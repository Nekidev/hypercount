//! An atomic, lock-free, hash map-like counter structure.
//!
//! It uses Robin Hood hashing for collision resolution and supports concurrent access and
//! modification of counters associated with keys. None of the operations require locks, making it
//! suitable for high-performance scenarios.
//!
//! # Features
//!
//! - Atomic operations for incrementing and decrementing counters.
//! - Lock-free design for concurrent access.
//! - Robin Hood hashing for efficient collision resolution.
//! - Automatic resizing of the underlying storage when needed.
//!
//! ## Notes Before Use
//!
//! - This is an experimental library and may not be suitable for production use yet. The API may
//!   change.
//! - It does not support shrinking of the underlying storage yet. However, since the structure
//!   uses a bucket of pointers, unused preallocated pointers only consume the null pointer size.
//! - ABA protection for pointer operations is not implemented yet.
//! - Performance optimizations are still ongoing.
//! - Operations on atomics are always wrapping on overflow.
//! - Since the map relies on atomic pointers before the atomic values, operations like `fetch_add`
//!   are not true read-modify-write operations. This may affect the choice of memory orderings.
//!   In those cases, the ordering is sanitized for loads and stores accordingly.
//!
//! # Getting Started
//!
//! To install this library, run the following command:
//!
//! ```sh
//! cargo add hypercounter
//! ```
//!
//! That's it! To start using it, create a new `HyperCounter` instance:
//!
//! ```rust
//! use std::sync::atomic::{AtomicUsize, Ordering};
//! use hypercounter::HyperCounter;
//!
//! let counter: HyperCounter<String, AtomicUsize> = HyperCounter::new();
//!
//! counter.fetch_add("example_key".to_string(), 1, Ordering::Relaxed);
//! counter.fetch_sub("example_key".to_string(), 1, Ordering::Relaxed);
//! ```
//!
//! Keys are automatically removed when their associated counter reaches zero. Neither inserts nor
//! removals are needed explicitly. If you want to remove a key manually, however, you can do so
//! using [`HyperCounter::swap()`] to swap the value with 0.
//!
//! ```rust
//! # use std::sync::atomic::{AtomicUsize, Ordering};
//! # use hypercounter::HyperCounter;
//! # let counter: HyperCounter<String, AtomicUsize> = HyperCounter::new();
//! let previous_value = counter.swap("example_key".to_string(), 0, Ordering::Relaxed);
//! ```
//!
//! ## Supported Operations
//!
//! The following atomic operations are supported:
//!
//! - [`HyperCounter::load()`]: Atomically loads the current value for a given key.
//! - [`HyperCounter::swap()`]: Atomically swaps the value for a given key.
//! - [`HyperCounter::fetch_add()`]: Atomically adds a value to the counter for a given key.
//! - [`HyperCounter::fetch_sub()`]: Atomically subtracts a value from the counter for a given key.
//! - [`HyperCounter::fetch_and()`]: Atomically performs a bitwise AND operation on the counter for
//!   a given key.
//! - [`HyperCounter::fetch_nand()`]: Atomically performs a bitwise NAND operation on the counter
//!   for a given key.
//! - [`HyperCounter::fetch_or()`]: Atomically performs a bitwise OR operation on the counter for a
//!   given key.
//! - [`HyperCounter::fetch_xor()`]: Atomically performs a bitwise XOR operation on the counter for
//!   a given key.
//! - [`HyperCounter::fetch_max()`]: Atomically sets the counter for a given key to the maximum of
//!   the current value and the provided value.
//! - [`HyperCounter::fetch_min()`]: Atomically sets the counter for a given key to the minimum of
//!   the current value and the provided value.
//!
//! ## Resizing
//!
//! The `HyperCounter` automatically duplicates its internal storage when the load factor exceeds
//! 75%. Its capacity starts at 8 and doubles with each expansion. Shrinking is not supported yet.
//!
//! # Benchmarking
//!
//! There's a simple benchmark example included in the `examples` directory. You can run it using:
//!
//! ```sh
//! cargo run --example bench
//! ```
//!
//! This will execute a series of single-threaded benchmarks and print the operations per second
//! for various scenarios.

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use crate::{numbers::AtomicNumber, pointer::SafeAtomicPtr};

mod numbers;
mod pointer;

type Bucket<K, V> = Vec<Arc<SafeAtomicPtr<(K, V)>>>;

/// Errors that can occur during Robin Hood insertion.
///
/// The end user does not need to handle these errors directly, as they are managed internally by
/// [`HyperCounter`].
///
/// Type Parameters:
/// * `K` - The key type.
/// * `V` - The value type.
#[derive(Debug)]
enum RobinHoodError<K, V> {
    /// A duplicate key was found.
    Duplicate(Arc<(K, V)>),

    /// The bucket is full (and currently expanding).
    Full,
}

pub struct HyperCounter<K, V, H = DefaultHasher>
where
    K: Eq + Hash,
    V: AtomicNumber,
    H: Hasher + Default,
{
    bucket: SafeAtomicPtr<Bucket<K, V>>,
    length: AtomicUsize,
    is_expanding: AtomicBool,
    hasher: PhantomData<H>,
}

impl<K, V, H> HyperCounter<K, V, H>
where
    K: Eq + Hash,
    V: AtomicNumber,
    H: Hasher + Default,
{
    /// Creates a new, empty HyperCounter.
    ///
    /// Returns:
    /// * [`HyperCounter<K, V>`] - A new HyperCounter instance.
    pub fn new() -> Self {
        Self {
            bucket: SafeAtomicPtr::new(Arc::new(Vec::new())),
            length: AtomicUsize::new(0),
            is_expanding: AtomicBool::new(false),
            hasher: PhantomData,
        }
    }

    /// Returns the current amount of occupied entries in the HyperCounter.
    ///
    /// Returns:
    /// * [`usize`] - The current length.
    pub fn len(&self) -> usize {
        self.length.load(Ordering::Acquire)
    }

    /// Checks if the HyperCounter is empty.
    ///
    /// Returns:
    /// * `true` - if the HyperCounter is empty.
    /// * `false` - otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the current capacity of the HyperCounter.
    ///
    /// Returns:
    /// * [`usize`] - The current capacity.
    pub fn capacity(&self) -> usize {
        self.bucket
            .load(Ordering::Acquire)
            .expect(
                "HyperCounter's bucket pointer was null. It should never be null! (this is a bug)",
            )
            .capacity()
    }

    /// Attempts to lease the resizing lock.
    ///
    /// Returns:
    /// * `true` - if the lease was successful.
    /// * `false` - if another thread is already resizing.
    fn lease_expansion(&self) -> Option<ExpansionLeaseGuard<'_, K, V, H>> {
        if self
            .is_expanding
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            Some(ExpansionLeaseGuard::new(self))
        } else {
            None
        }
    }

    /// Determines if the bucket should be expanded.
    ///
    /// It returns `true` if the load factor exceeds 75%.
    ///
    /// Arguments:
    /// * `bucket` - The current bucket to evaluate.
    ///
    /// Returns:
    /// * `true` - if the bucket should be expanded.
    /// * `false` - otherwise.
    fn should_expand(&self, bucket: &Bucket<K, V>) -> bool {
        bucket.is_empty() || self.len() * 100 / bucket.len() >= 75
    }

    /// Inserts an entry into the bucket using Robin Hood hashing.
    ///
    /// Arguments:
    /// * `bucket` - The bucket to insert into.
    /// * `entry` - The entry to insert.
    ///
    /// Returns:
    /// * `Ok(())` - if the insertion was successful.
    /// * `Err(Arc<(K, V)>)` - if a duplicate key was found, containing the existing entry.
    fn robin_hood_insert(
        &self,
        bucket: &Bucket<K, V>,
        entry: Arc<(K, V)>,
    ) -> Result<(), RobinHoodError<K, V>> {
        let mut current = entry;

        let capacity = bucket.len();

        if capacity == 0 {
            return Err(RobinHoodError::Full);
        }

        let hash = self.hash(&current.0);
        let index = hash as usize % capacity;

        let mut i = 0;

        while i < capacity {
            let slot_index = (index + i) % capacity;
            let slot = &bucket[slot_index];

            // If it fails, it'll be `Some`.
            if let Err(Some(existing)) = slot.compare_exchange(
                None,
                Some(current.clone()),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                if current.0 == existing.0 {
                    return Err(RobinHoodError::Duplicate(existing));
                }

                let hash_current = self.hash(&current.0);
                let hash_existing = self.hash(&existing.0);

                let index_current = hash_current as usize % capacity;
                let index_existing = hash_existing as usize % capacity;

                let distance_current = (slot_index + capacity - index_current) % capacity;
                let distance_existing = (slot_index + capacity - index_existing) % capacity;

                if distance_current > distance_existing {
                    if slot
                        .compare_exchange(
                            Some(&existing),
                            Some(current.clone()),
                            Ordering::AcqRel,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        current = existing;

                        i += 1;
                    } else {
                        // Couldn't swap because the pointer's value changed, try again.
                        // That's why no i += 1 here.
                    }
                } else {
                    i += 1;
                }
            } else {
                return Ok(());
            }
        }

        Err(RobinHoodError::Full)
    }

    /// Removes an entry from the bucket using Robin Hood backward shifting.
    ///
    /// Arguments:
    /// * `bucket` - The bucket to remove from.
    /// * `key` - The key to remove.
    ///
    /// Returns:
    /// * `Some(Arc<(K, V)>)` - if the entry was found and removed.
    /// * `None` - if the entry was not found.
    fn robin_hood_remove(&self, bucket: &Bucket<K, V>, key: &K) -> Option<Arc<(K, V)>> {
        let hash = self.hash(key);
        let capacity = bucket.len();

        if capacity == 0 {
            return None;
        }

        let origin = hash as usize % capacity;
        let mut i = origin;

        let entry = loop {
            let slot = &bucket[i];
            let loaded = slot.load(Ordering::Acquire);

            if let Some(entry) = loaded {
                if entry.0 == *key {
                    break entry;
                }

                let key_dib = if i >= origin {
                    i - origin
                } else {
                    i + capacity - origin
                };

                let entry_hash = self.hash(&entry.0);
                let entry_index = entry_hash as usize % capacity;

                let entry_dib = if i >= entry_index {
                    i - entry_index
                } else {
                    i + capacity - entry_index
                };

                if entry_dib < key_dib {
                    // The found key was closer to its origin than the searched key, so the
                    // searched key cannot be in the table.
                    return None;
                }

                i += 1;
                i %= capacity;

                if i == origin {
                    // Wrapped around the bucket and didn't find the key.
                    return None;
                }
            }
        };

        let mut prev_entry = entry.clone();

        let mut iter = 0;

        // Now perform backward shifting to fill the gap.
        while iter < capacity {
            let curr_index = (i + 1) % capacity;
            let prev_index = i % capacity;

            let curr_slot = &bucket[curr_index];
            let curr = curr_slot.load(Ordering::Acquire);

            if let Some(curr_entry) = curr {
                let curr_entry_index = self.hash(&curr_entry.0) as usize % capacity;

                if curr_entry_index == curr_index {
                    // In its original position, stop shifting.
                    break;
                }

                let prev_slot = &bucket[prev_index];

                let swap_result = prev_slot.compare_exchange(
                    Some(&prev_entry),
                    Some(curr_entry),
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                );

                match swap_result {
                    Ok(Some(prev_curr_entry)) => {
                        prev_entry = prev_curr_entry;
                        i += 1;
                        i %= capacity;
                        iter += 1;
                    }
                    Ok(None) => {
                        // This should not happen, as we loaded it just before. If it had been
                        // changed to None after load, the swap would have failed instead.
                        break;
                    }
                    Err(Some(prev_curr_entry)) => {
                        // Couldn't swap because the pointer's value changed, try again.
                        prev_entry = prev_curr_entry;
                        continue;
                    }
                    Err(None) => {
                        // The slot became empty, stop shifting.
                        break;
                    }
                }
            } else {
                break;
            }
        }

        let slot = &bucket[i];
        slot.compare_exchange(Some(&prev_entry), None, Ordering::AcqRel, Ordering::Relaxed)
            .ok();

        Some(entry)
    }

    /// Expands the bucket to a new capacity.
    ///
    /// Capacity is doubled, or set to 8 if the bucket is empty.
    ///
    /// Arguments:
    /// * `bucket` - The current bucket to expand.
    /// * `_lease` - The expansion lease guard.
    ///
    /// Returns:
    /// * [`Arc<Bucket<K, V>>`] - The new expanded bucket.
    fn expand(
        &self,
        bucket: &Bucket<K, V>,
        _lease: ExpansionLeaseGuard<'_, K, V, H>,
    ) -> Arc<Bucket<K, V>> {
        let new_capacity = if bucket.is_empty() {
            8
        } else {
            bucket.len() * 2
        };

        let new_bucket: Vec<Arc<SafeAtomicPtr<(K, V)>>> = (0..new_capacity)
            .map(|_| Arc::new(SafeAtomicPtr::null()))
            .collect();

        for entry in bucket {
            if let Some(entry) = entry.load(Ordering::Acquire) {
                // TODO: Is this really safe to ignore?
                let _ = self.robin_hood_insert(&new_bucket, entry);
            }
        }

        let new_bucket = Arc::new(new_bucket);

        self.bucket
            .swap(Some(new_bucket.clone()), Ordering::Release);

        new_bucket
    }

    /// Gets the appropriate load ordering based on the provided ordering.
    ///
    /// This is because [`HyperCounter::fetch_add()`] is not a real read-modify-write operation
    /// since it has to combine a fetch to the atomic pointer before updating the atomic number.
    ///
    /// Arguments:
    /// * `ordering` - The original memory ordering.
    ///
    /// Returns:
    /// * `Ordering` - The adjusted load ordering.
    fn get_load_ordering(&self, ordering: Ordering) -> Ordering {
        match ordering {
            Ordering::Release | Ordering::AcqRel => Ordering::Acquire,
            o => o,
        }
    }

    /// Gets the appropriate store ordering based on the provided ordering.
    ///
    /// This is because [`HyperCounter::fetch_add()`] is not a real read-modify-write operation
    /// since it has to combine a fetch to the atomic pointer before updating the atomic number.
    ///
    /// Arguments:
    /// * `ordering` - The original memory ordering.
    ///
    /// Returns:
    /// * `Ordering` - The adjusted store ordering.
    fn get_store_ordering(&self, ordering: Ordering) -> Ordering {
        match ordering {
            Ordering::Acquire | Ordering::AcqRel => Ordering::Release,
            o => o,
        }
    }

    /// Fetches an entry by key.
    ///
    /// Arguments:
    /// * `key` - The key to fetch.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * `Some(Arc<(K, V)>)` - if the entry was found.
    /// * `None` - if the entry was not found.
    fn fetch(&self, key: &K, ordering: Ordering) -> Option<Arc<(K, V)>> {
        let hash = self.hash(key);

        let bucket = self
            .bucket
            .load(ordering)
            .expect("The HyperCounter bucket pointer was null! It should never be null.");

        if bucket.is_empty() {
            return None;
        }

        for i in 0..bucket.len() {
            let index = (hash as usize + i) % bucket.len();

            // Return None if the entry is vacant
            let entry_ptr = bucket[index].load(ordering)?;

            if entry_ptr.0 == *key {
                return Some(entry_ptr);
            }

            // TODO: Check DIB. If DIB < i, break early and return None.
        }

        None
    }

    /// Inserts a new entry into the [`HyperCounter`].
    ///
    /// It will increase the length counter on successful insertion.
    ///
    /// Arguments:
    /// * `pair` - The key-value pair to insert.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * `Ok(())` - if the insertion was successful.
    /// * `Err(RobinHoodError<K, V>)` - if an error occurred during insertion.
    fn insert(&self, pair: Arc<(K, V)>, ordering: Ordering) -> Result<(), RobinHoodError<K, V>> {
        let mut bucket = self
            .bucket
            .load(self.get_load_ordering(ordering))
            .expect("The HyperCounter bucket pointer was null! It should never be null.");

        if self.should_expand(&bucket)
            && let Some(lease) = self.lease_expansion()
        {
            bucket = self.expand(&bucket, lease);
        }

        self.robin_hood_insert(&bucket, pair)?;
        self.length.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Removes an entry by key.
    ///
    /// It will decrease the length counter on successful removal.
    ///
    /// Arguments:
    /// * `key` - The key to remove.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * `Some(Arc<(K, V)>)` - if the entry was found and removed.
    /// * `None` - if the entry was not found.
    fn remove(&self, key: &K, ordering: Ordering) -> Option<Arc<(K, V)>> {
        let bucket = self
            .bucket
            .load(self.get_load_ordering(ordering))
            .expect("The HyperCounter bucket pointer was null! It should never be null.");

        let result = self.robin_hood_remove(&bucket, key)?;
        self.length.fetch_sub(1, Ordering::Relaxed);

        Some(result)
    }

    /// Hashes a key to a [`u64`] value.
    ///
    /// Arguments:
    /// * `key` - The key to hash.
    ///
    /// Returns:
    /// * [`u64`] - The hashed value.
    fn hash(&self, key: &K) -> u64 {
        let mut hasher = H::default();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Atomically loads the value for the given key.
    ///
    /// Arguments:
    /// * `key` - The key to load.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * [`V::Primitive`] - The current value associated with the key, or zero if the key is
    ///   missing.
    pub fn load(&self, key: &K, ordering: Ordering) -> V::Primitive {
        self.fetch(key, ordering)
            .map(|i| i.1.load(ordering))
            .unwrap_or(V::ZERO)
    }

    /// Atomically swaps the value for the given key.
    ///
    /// If the new value is zero, the entry is removed.
    ///
    /// If the key is missing, a new entry is created with the new value.
    ///
    /// Arguments:
    /// * `key` - The key to swap.
    /// * `new_value` - The new value to set.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * [`V::Primitive`] - The previous value associated with the key. (before swap)
    pub fn swap(&self, key: K, new_value: V::Primitive, ordering: Ordering) -> V::Primitive {
        if new_value == V::ZERO {
            let old = self.remove(&key, self.get_store_ordering(ordering));

            old.map(|i| i.1.load(self.get_load_ordering(ordering)))
                .unwrap_or(V::ZERO)
        } else {
            let new_entry = Arc::new((key, V::new(new_value)));

            loop {
                let entry = self.fetch(&new_entry.0, self.get_load_ordering(ordering));

                if let Some(entry) = entry {
                    return entry.1.swap(new_value, ordering);
                } else {
                    match self.insert(new_entry.clone(), self.get_store_ordering(ordering)) {
                        Ok(()) => {
                            return V::ZERO;
                        }
                        Err(RobinHoodError::Duplicate(existing)) => {
                            // Another thread inserted it first, insert to that value instead.
                            let entry = existing;
                            return entry.1.swap(new_value, ordering);
                        }
                        Err(RobinHoodError::Full) => {
                            // The bucket is full, try again (after expansion).
                            continue;
                        }
                    }
                }
            }
        }
    }

    /// Atomically adds a value to the counter for the given key.
    ///
    /// If the key ends up being zero after addition, the entry is removed.
    ///
    /// If the key is missing, a new entry is created with the given value.
    ///
    /// Arguments:
    /// * `key` - The key to add to.
    /// * `value` - The value to add.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * [`V::Primitive`] - The previous value associated with the key. (before addition)
    pub fn fetch_add(&self, key: K, value: V::Primitive, ordering: Ordering) -> V::Primitive {
        let new_entry = Arc::new((key, V::new(value)));

        loop {
            let entry = self.fetch(&new_entry.0, self.get_load_ordering(ordering));

            if let Some(entry) = entry {
                // TODO: If the value after fetch_add() is zero, remove the entry.
                let result = entry.1.fetch_add(value, ordering);

                if V::primitive_wrapping_add(result, value) == V::ZERO {
                    self.remove(&entry.0, self.get_store_ordering(ordering));
                }

                return result;
            } else {
                let result = self.insert(new_entry.clone(), self.get_store_ordering(ordering));

                if let Err(RobinHoodError::Duplicate(existing)) = result {
                    // Another thread inserted it first, insert to that value instead.
                    return existing.1.fetch_add(value, ordering);
                } else if let Err(RobinHoodError::Full) = result {
                    // The bucket is full, try again (after expansion).
                    continue;
                }

                return V::ZERO;
            }
        }
    }

    /// Atomically subtracts a value from the counter for the given key.
    ///
    /// If the key ends up being zero after subtraction, the entry is removed.
    ///
    /// If the key is missing, a new entry is created with zero - the given value.
    ///
    /// Arguments:
    /// * `key` - The key to subtract from.
    /// * `value` - The value to subtract.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * [`V::Primitive`] - The previous value associated with the key. (before subtraction)
    pub fn fetch_sub(&self, key: K, value: V::Primitive, ordering: Ordering) -> V::Primitive {
        let new_entry = Arc::new((key, V::new(V::primitive_wrapping_sub(V::ZERO, value))));

        loop {
            let entry = self.fetch(&new_entry.0, self.get_load_ordering(ordering));

            if let Some(entry) = entry {
                // TODO: If the value after fetch_sub() is zero, remove the entry.
                let result = entry.1.fetch_sub(value, ordering);

                if V::primitive_wrapping_sub(result, value) == V::ZERO {
                    self.remove(&entry.0, self.get_store_ordering(ordering));
                }

                return result;
            } else {
                let result = self.insert(new_entry.clone(), self.get_store_ordering(ordering));

                if let Err(RobinHoodError::Duplicate(existing)) = result {
                    // Another thread inserted it first, insert to that value instead.
                    return existing.1.fetch_sub(value, ordering);
                } else if let Err(RobinHoodError::Full) = result {
                    // The bucket is full, try again (after expansion).
                    continue;
                }

                return V::ZERO;
            }
        }
    }

    /// Atomically performs a bitwise AND operation on the counter for the given key.
    ///
    /// If the key is missing, nothing is done and zero is returned.
    ///
    /// Arguments:
    /// * `key` - The key to perform the AND operation on.
    /// * `value` - The value to AND with.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * [`V::Primitive`] - The previous value associated with the key. (before AND operation)
    pub fn fetch_and(&self, key: &K, value: V::Primitive, ordering: Ordering) -> V::Primitive {
        let entry = self.fetch(key, self.get_load_ordering(ordering));

        if let Some(entry) = entry {
            entry.1.fetch_and(value, ordering)
        } else {
            V::ZERO
        }
    }

    /// Atomically performs a bitwise NAND operation on the counter for the given key.
    ///
    /// If the key is missing, the new value is inserted and all bits set (i.e., !0) is returned.
    ///
    /// Arguments:
    /// * `key` - The key to perform the NAND operation on.
    /// * `value` - The value to NAND with.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * [`V::Primitive`] - The previous value associated with the key. (before NAND operation)
    pub fn fetch_nand(&self, key: K, value: V::Primitive, ordering: Ordering) -> V::Primitive {
        let new_entry = Arc::new((key, V::new(!value)));

        loop {
            let entry = self.fetch(&new_entry.0, self.get_load_ordering(ordering));

            if let Some(entry) = entry {
                let result = entry.1.fetch_nand(value, ordering);

                if !(result & value) == V::ZERO {
                    self.remove(&entry.0, self.get_store_ordering(ordering));
                }

                return result;
            } else {
                let result = self.insert(new_entry.clone(), self.get_store_ordering(ordering));

                if let Err(RobinHoodError::Duplicate(existing)) = result {
                    // Another thread inserted it first, insert to that value instead.
                    return existing.1.fetch_nand(value, ordering);
                } else if let Err(RobinHoodError::Full) = result {
                    // The bucket is full, try again (after expansion).
                    continue;
                }

                return !V::ZERO;
            }
        }
    }

    /// Atomically performs a bitwise OR operation on the counter for the given key.
    ///
    /// If the key is missing, the new value is inserted and zero is returned.
    ///
    /// Arguments:
    /// * `key` - The key to perform the OR operation on.
    /// * `value` - The value to OR with.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * [`V::Primitive`] - The previous value associated with the key. (before OR operation)
    pub fn fetch_or(&self, key: K, value: V::Primitive, ordering: Ordering) -> V::Primitive {
        let new_entry = Arc::new((key, V::new(value)));

        loop {
            let entry = self.fetch(&new_entry.0, self.get_load_ordering(ordering));

            if let Some(entry) = entry {
                let result = entry.1.fetch_or(value, ordering);

                if result | value == V::ZERO {
                    self.remove(&entry.0, self.get_store_ordering(ordering));
                }

                return result;
            } else {
                let result = self.insert(new_entry.clone(), self.get_store_ordering(ordering));

                if let Err(RobinHoodError::Duplicate(existing)) = result {
                    // Another thread inserted it first, insert to that value instead.
                    return existing.1.fetch_or(value, ordering);
                } else if let Err(RobinHoodError::Full) = result {
                    // The bucket is full, try again (after expansion).
                    continue;
                }

                return V::ZERO;
            }
        }
    }

    /// Atomically performs a bitwise XOR operation on the counter for the given key.
    ///
    /// If the key is missing, the new value is inserted and zero is returned.
    ///
    /// If the resulting value is zero after the XOR operation, the entry is removed.
    ///
    /// Arguments:
    /// * `key` - The key to perform the XOR operation on.
    /// * `value` - The value to XOR with.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * [`V::Primitive`] - The previous value associated with the key. (before XOR operation)
    pub fn fetch_xor(&self, key: K, value: V::Primitive, ordering: Ordering) -> V::Primitive {
        let new_entry = Arc::new((key, V::new(value)));

        loop {
            let entry = self.fetch(&new_entry.0, self.get_load_ordering(ordering));

            if let Some(entry) = entry {
                let result = entry.1.fetch_xor(value, ordering);

                if result ^ value == V::ZERO {
                    self.remove(&entry.0, self.get_store_ordering(ordering));
                }

                return result;
            } else {
                let result = self.insert(new_entry.clone(), self.get_store_ordering(ordering));

                if let Err(RobinHoodError::Duplicate(existing)) = result {
                    // Another thread inserted it first, insert to that value instead.
                    return existing.1.fetch_xor(value, ordering);
                } else if let Err(RobinHoodError::Full) = result {
                    // The bucket is full, try again (after expansion).
                    continue;
                }

                return V::ZERO;
            }
        }
    }

    /// Atomically sets the counter for the given key to the maximum of its current value and the
    /// given value.
    ///
    /// If the key is missing and the value is higher than zero, the new value is inserted and zero
    /// is returned. Otherwise, nothing is done and zero is returned.
    ///
    /// Arguments:
    /// * `key` - The key to perform the max operation on.
    /// * `value` - The value to compare with.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * [`V::Primitive`] - The previous value associated with the key. (before max operation)
    pub fn fetch_max(&self, key: K, value: V::Primitive, ordering: Ordering) -> V::Primitive {
        let new_entry = Arc::new((key, V::new(value)));

        loop {
            let entry = self.fetch(&new_entry.0, self.get_load_ordering(ordering));

            if let Some(entry) = entry {
                let result = entry.1.fetch_max(value, ordering);

                if value >= result && value == V::ZERO {
                    self.remove(&entry.0, self.get_store_ordering(ordering));
                }

                return result;
            } else {
                if value <= V::ZERO {
                    return V::ZERO;
                }

                let result = self.insert(new_entry.clone(), self.get_store_ordering(ordering));

                if let Err(RobinHoodError::Duplicate(existing)) = result {
                    // Another thread inserted it first, insert to that value instead.
                    return existing.1.fetch_max(value, ordering);
                } else if let Err(RobinHoodError::Full) = result {
                    // The bucket is full, try again (after expansion).
                    continue;
                }

                return V::ZERO;
            }
        }
    }

    /// Atomically sets the counter for the given key to the minimum of its current value and the
    /// given value.
    ///
    /// If the key is missing and the value is lower than zero, the new value is inserted and zero
    /// is returned. Otherwise, nothing is done and zero is returned.
    ///
    /// Arguments:
    /// * `key` - The key to perform the min operation on.
    /// * `value` - The value to compare with.
    /// * `ordering` - The memory ordering to use.
    ///
    /// Returns:
    /// * [`V::Primitive`] - The previous value associated with the key. (before min operation)
    pub fn fetch_min(&self, key: K, value: V::Primitive, ordering: Ordering) -> V::Primitive {
        let new_entry = Arc::new((key, V::new(value)));

        loop {
            let entry = self.fetch(&new_entry.0, self.get_load_ordering(ordering));

            if let Some(entry) = entry {
                let result = entry.1.fetch_min(value, ordering);

                if value <= result && value == V::ZERO {
                    self.remove(&entry.0, self.get_store_ordering(ordering));
                }

                return result;
            } else {
                if value >= V::ZERO {
                    return V::ZERO;
                }

                let result = self.insert(new_entry.clone(), self.get_store_ordering(ordering));

                if let Err(RobinHoodError::Duplicate(existing)) = result {
                    // Another thread inserted it first, insert to that value instead.
                    return existing.1.fetch_min(value, ordering);
                } else if let Err(RobinHoodError::Full) = result {
                    // The bucket is full, try again (after expansion).
                    continue;
                }

                return V::ZERO;
            }
        }
    }
}

impl<K, V> Default for HyperCounter<K, V>
where
    K: Eq + Hash,
    V: AtomicNumber,
{
    fn default() -> Self {
        Self::new()
    }
}

struct ExpansionLeaseGuard<'a, K, V, H>
where
    K: Eq + Hash,
    V: AtomicNumber,
    H: Hasher + Default,
{
    hypercounter: &'a HyperCounter<K, V, H>,
}

impl<'a, K, V, H> ExpansionLeaseGuard<'a, K, V, H>
where
    K: Eq + Hash,
    V: AtomicNumber,
    H: Hasher + Default,
{
    fn new(hypercounter: &'a HyperCounter<K, V, H>) -> Self {
        Self { hypercounter }
    }
}

impl<'a, K, V, H> Drop for ExpansionLeaseGuard<'a, K, V, H>
where
    K: Eq + Hash,
    V: AtomicNumber,
    H: Hasher + Default,
{
    fn drop(&mut self) {
        self.hypercounter
            .is_expanding
            .store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_hypercounter_basic() {
        let counter: HyperCounter<String, AtomicUsize> = HyperCounter::new();

        assert_eq!(counter.len(), 0);
        assert!(counter.is_empty());

        let prev = counter.fetch_add("apple".to_string(), 5, Ordering::SeqCst);
        assert_eq!(prev, 0);
        assert_eq!(counter.len(), 1);
        assert!(!counter.is_empty());
        assert_eq!(counter.capacity(), 8);

        let prev = counter.fetch_add("apple".to_string(), 3, Ordering::SeqCst);
        assert_eq!(prev, 5);

        let prev = counter.fetch_add("banana".to_string(), 2, Ordering::SeqCst);
        assert_eq!(prev, 0);
        assert_eq!(counter.len(), 2);

        let prev = counter.fetch_sub("apple".to_string(), 8, Ordering::SeqCst);
        assert_eq!(prev, 8);
        let load = counter.load(&"apple".to_string(), Ordering::SeqCst);
        assert_eq!(load, 0);
        assert_eq!(counter.len(), 1); // "apple" should be removed

        let prev = counter.fetch_sub("banana".to_string(), 2, Ordering::SeqCst);
        assert_eq!(prev, 2);
        assert_eq!(counter.len(), 0); // "banana" should be removed
    }

    #[test]
    fn test_hypercounter_expand() {
        let counter: HyperCounter<usize, AtomicUsize> = HyperCounter::new();

        for i in 0..100 {
            counter.fetch_add(i, i, Ordering::SeqCst);
            assert_eq!(counter.len(), i + 1);
        }

        assert_eq!(counter.len(), 100);

        for i in 0..100 {
            let load = counter.load(&i, Ordering::SeqCst);
            assert_eq!(load, i);
        }
    }

    #[test]
    fn test_hypercounter_remove() {
        let counter: HyperCounter<usize, AtomicUsize> = HyperCounter::new();

        for i in 0..100 {
            counter.fetch_add(i, i, Ordering::SeqCst);
        }

        for i in 0..100 {
            let prev = counter.fetch_sub(i, i, Ordering::SeqCst);
            assert_eq!(prev, i);
            let load = counter.load(&i, Ordering::SeqCst);
            assert_eq!(load, 0);

            assert_eq!(counter.len(), 99 - i);
        }
    }

    #[test]
    fn test_hypercounter_orderings() {
        let counter: HyperCounter<String, AtomicUsize> = HyperCounter::new();

        let orderings = [
            Ordering::AcqRel,
            Ordering::Acquire,
            Ordering::Release,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ];

        for &ordering in &orderings {
            counter.fetch_add("key".to_string(), 10, ordering);
            counter.fetch_sub("key".to_string(), 5, ordering);
            counter.fetch_and(&"key".to_string(), 7, ordering);
            counter.fetch_nand("key".to_string(), 3, ordering);
            counter.fetch_or("key".to_string(), 12, ordering);
            counter.fetch_xor("key".to_string(), 6, ordering);
            counter.fetch_max("key".to_string(), 15, ordering);
            counter.fetch_min("key".to_string(), 5, ordering);
            counter.swap("key".to_string(), 20, ordering);
        }

        let load_orderings = [Ordering::Acquire, Ordering::SeqCst, Ordering::Relaxed];

        for &ordering in &load_orderings {
            counter.load(&"key".to_string(), ordering);
        }
    }
}
