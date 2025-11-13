//! An atomic hash map implementation using Robin Hood hashing.
//! 
//! This module provides the [`TurboMap`] struct, which is a concurrent hash map that uses Robin
//! Hood hashing for collision resolution. It supports atomic insertions, removals, and lookups,
//! making it suitable for multi-threaded environments.
//! 
//! It backs [`HyperCounter`](crate::HyperCounter) to provide efficient storage for per-key
//! counters.
//! 
//! # Usage
//! 
//! Using [`TurboMap`] is straightforward. You can create a new instance, insert key-value pairs,
//! retrieve values by key, and remove entries.
//! 
//! ```rust
//! use hypercounter::map::TurboMap;
//! use std::sync::Arc;
//! 
//! let map: TurboMap<String, i32> = TurboMap::new();
//! map.insert(Arc::new(("key".to_string(), 42))).unwrap();
//! 
//! let value = map.get(&"key".to_string()).unwrap();
//! assert_eq!(value.1, 42);
//! 
//! map.remove(&"key".to_string());
//! assert!(map.get(&"key".to_string()).is_none());
//! ```

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicIsize, Ordering},
    },
};

use arc_swap::{ArcSwap, ArcSwapOption};

type Bucket<K, V> = Vec<Arc<ArcSwapOption<(K, V)>>>;

/// Errors that can occur during Robin Hood insertion.
///
/// The end user does not need to handle these errors directly, as they are managed internally by
/// [`TurboMap`].
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

pub struct TurboMap<K, V, H = DefaultHasher>
where
    K: Eq + Hash,
    H: Hasher + Default,
{
    bucket: ArcSwap<Bucket<K, V>>,
    length: AtomicIsize,
    is_expanding: AtomicBool,
    hasher: PhantomData<H>,
}

impl<K, V, H> TurboMap<K, V, H>
where
    K: Eq + Hash,
    H: Hasher + Default,
{
    /// Creates a new, empty [`TurboMap`].
    ///
    /// Returns:
    /// * [`TurboMap<K, V>`] - A new [`TurboMap`] instance.
    pub fn new() -> Self {
        Self {
            bucket: ArcSwap::new(Arc::new(Vec::new())),
            length: AtomicIsize::new(0),
            is_expanding: AtomicBool::new(false),
            hasher: PhantomData,
        }
    }

    /// Returns the current amount of occupied entries in the [`TurboMap`].
    ///
    /// Returns:
    /// * [`usize`] - The current length.
    pub fn len(&self) -> usize {
        let value = self.length.load(Ordering::Acquire);

        if value < 0 { 0 } else { value as usize }
    }

    /// Checks if the [`TurboMap`] is empty.
    ///
    /// Returns:
    /// * `true` - if the [`TurboMap`] is empty.
    /// * `false` - otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the current capacity of the [`TurboMap`].
    ///
    /// Returns:
    /// * [`usize`] - The current capacity.
    pub fn capacity(&self) -> usize {
        self.bucket.load().capacity()
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

            let result = slot.compare_and_swap(&None::<Arc<(K, V)>>, Some(current.clone()));

            // If it fails, it'll be `Some` because we compared against `None`.
            if let Some(existing) = &*result {
                let existing = existing.clone();

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
                    let option_existing = Some(existing.clone());

                    if slot
                        .compare_and_swap(&option_existing, Some(current.clone()))
                        .as_ref()
                        .map(Arc::as_ptr)
                        == option_existing.as_ref().map(Arc::as_ptr)
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
            let loaded = slot.load();

            if let Some(entry) = loaded.clone() {
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
            let curr = curr_slot.load();

            if let Some(curr_entry) = &*curr {
                let curr_entry_index = self.hash(&curr_entry.0) as usize % capacity;

                if curr_entry_index == curr_index {
                    // In its original position, stop shifting.
                    break;
                }

                let prev_slot = &bucket[prev_index];

                let swap_result =
                    prev_slot.compare_and_swap(&Some(prev_entry.clone()), Some(curr_entry.clone()));

                let swap_result = {
                    let success =
                        swap_result.as_ref().map(Arc::as_ptr) == Some(Arc::as_ptr(&prev_entry));

                    if success {
                        Ok(swap_result.clone())
                    } else {
                        Err(swap_result.clone())
                    }
                };

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
        slot.compare_and_swap(&Some(prev_entry), None);

        Some(entry.clone())
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

        let new_bucket: Vec<Arc<ArcSwapOption<(K, V)>>> = (0..new_capacity)
            .map(|_| Arc::new(ArcSwapOption::empty()))
            .collect();

        for entry in bucket {
            if let Some(entry) = entry.load().clone() {
                // TODO: Is this really safe to ignore?
                let _ = self.robin_hood_insert(&new_bucket, entry);
            }
        }

        let new_bucket = Arc::new(new_bucket);

        self.bucket.swap(new_bucket.clone());

        new_bucket
    }

    /// Gets an entry by key.
    ///
    /// Arguments:
    /// * `key` - The key to fetch.
    ///
    /// Returns:
    /// * `Some(Arc<(K, V)>)` - if the entry was found.
    /// * `None` - if the entry was not found.
    pub fn get(&self, key: &K) -> Option<Arc<(K, V)>> {
        let hash = self.hash(key);

        let bucket = self.bucket.load();

        if bucket.is_empty() {
            return None;
        }

        for i in 0..bucket.len() {
            let index = (hash as usize + i) % bucket.len();

            // Return None if the entry is vacant
            let entry_ptr = bucket[index].load().clone()?;

            if entry_ptr.0 == *key {
                return Some(entry_ptr);
            }

            // TODO: Check DIB. If DIB < i, break early and return None.
        }

        None
    }

    /// Inserts a new entry into the [`TurboMap`].
    ///
    /// It will increase the length counter on successful insertion.
    ///
    /// Arguments:
    /// * `pair` - The key-value pair to insert.
    ///
    /// Returns:
    /// * [`Ok<()>`] - If the insertion was successful.
    /// * [`Err<Arc<(K, V)>>`] - If the key was duplicate, the existing entry is returned.
    pub fn insert(&self, pair: Arc<(K, V)>) -> Result<(), Arc<(K, V)>> {
        let mut bucket = self.bucket.load().clone();

        if self.should_expand(&bucket)
            && let Some(lease) = self.lease_expansion()
        {
            bucket = self.expand(&bucket, lease);
        }

        loop {
            match self.robin_hood_insert(&bucket, pair.clone()) {
                Ok(()) => break,
                Err(RobinHoodError::Full) => {
                    if let Some(lease) = self.lease_expansion() {
                        bucket = self.expand(&bucket, lease);
                    }
                }
                Err(RobinHoodError::Duplicate(existing)) => return Err(existing),
            }
        }

        self.length.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Removes an entry by key.
    ///
    /// It will decrease the length counter on successful removal.
    ///
    /// Arguments:
    /// * `key` - The key to remove.
    ///
    /// Returns:
    /// * `Some(Arc<(K, V)>)` - if the entry was found and removed.
    /// * `None` - if the entry was not found.
    pub fn remove(&self, key: &K) -> Option<Arc<(K, V)>> {
        let bucket = self.bucket.load();

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
}

impl<K, V> Default for TurboMap<K, V>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

struct ExpansionLeaseGuard<'a, K, V, H>
where
    K: Eq + Hash,
    H: Hasher + Default,
{
    map: &'a TurboMap<K, V, H>,
}

impl<'a, K, V, H> ExpansionLeaseGuard<'a, K, V, H>
where
    K: Eq + Hash,
    H: Hasher + Default,
{
    fn new(map: &'a TurboMap<K, V, H>) -> Self {
        Self { map }
    }
}

impl<'a, K, V, H> Drop for ExpansionLeaseGuard<'a, K, V, H>
where
    K: Eq + Hash,
    H: Hasher + Default,
{
    fn drop(&mut self) {
        self.map.is_expanding.store(false, Ordering::Release);
    }
}
