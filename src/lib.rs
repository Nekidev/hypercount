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
//! - Performance optimizations are still ongoing.
//! - Operations on atomics are always wrapping on overflow.
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
    sync::{Arc, atomic::Ordering},
};

use crate::{map::TurboMap, numbers::AtomicNumber};

pub mod map;
mod numbers;

pub struct HyperCounter<K, V, H = DefaultHasher>
where
    K: Eq + Hash,
    V: AtomicNumber,
    H: Hasher + Default,
{
    inner: TurboMap<K, V, H>,
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
            inner: TurboMap::new(),
        }
    }

    /// Returns the current amount of occupied entries in the HyperCounter.
    ///
    /// Returns:
    /// * [`usize`] - The current length.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Checks if the HyperCounter is empty.
    ///
    /// Returns:
    /// * `true` - if the HyperCounter is empty.
    /// * `false` - otherwise.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the current capacity of the HyperCounter.
    ///
    /// Returns:
    /// * [`usize`] - The current capacity.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
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
        self.inner
            .get(key)
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
            let old = self.inner.remove(&key);

            old.map(|i| i.1.load(self.get_load_ordering(ordering)))
                .unwrap_or(V::ZERO)
        } else {
            let new_entry = Arc::new((key, V::new(new_value)));

            let entry = self.inner.get(&new_entry.0);

            if let Some(entry) = entry {
                entry.1.swap(new_value, ordering)
            } else {
                match self.inner.insert(new_entry.clone()) {
                    Ok(()) => V::ZERO,
                    Err(existing) => {
                        // Another thread inserted it first, insert to that value instead.
                        let entry = existing;
                        entry.1.swap(new_value, ordering)
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

        let entry = self.inner.get(&new_entry.0);

        if let Some(entry) = entry {
            // TODO: If the value after fetch_add() is zero, remove the entry.
            let result = entry.1.fetch_add(value, ordering);

            if V::primitive_wrapping_add(result, value) == V::ZERO {
                self.inner.remove(&entry.0);
            }

            result
        } else {
            let result = self.inner.insert(new_entry.clone());

            if let Err(existing) = result {
                // Another thread inserted it first, insert to that value instead.
                return existing.1.fetch_add(value, ordering);
            }

            V::ZERO
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

        let entry = self.inner.get(&new_entry.0);

        if let Some(entry) = entry {
            // TODO: If the value after fetch_sub() is zero, remove the entry.
            let result = entry.1.fetch_sub(value, ordering);

            if V::primitive_wrapping_sub(result, value) == V::ZERO {
                self.inner.remove(&entry.0);
            }

            result
        } else {
            let result = self.inner.insert(new_entry.clone());

            if let Err(existing) = result {
                // Another thread inserted it first, insert to that value instead.
                return existing.1.fetch_sub(value, ordering);
            }

            V::ZERO
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
        let entry = self.inner.get(key);

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

        let entry = self.inner.get(&new_entry.0);

        if let Some(entry) = entry {
            let result = entry.1.fetch_nand(value, ordering);

            if !(result & value) == V::ZERO {
                self.inner.remove(&entry.0);
            }

            result
        } else {
            let result = self.inner.insert(new_entry.clone());

            if let Err(existing) = result {
                // Another thread inserted it first, insert to that value instead.
                return existing.1.fetch_nand(value, ordering);
            }

            !V::ZERO
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

        let entry = self.inner.get(&new_entry.0);

        if let Some(entry) = entry {
            let result = entry.1.fetch_or(value, ordering);

            if result | value == V::ZERO {
                self.inner.remove(&entry.0);
            }

            result
        } else {
            let result = self.inner.insert(new_entry.clone());

            if let Err(existing) = result {
                // Another thread inserted it first, insert to that value instead.
                return existing.1.fetch_or(value, ordering);
            }

            V::ZERO
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

        let entry = self.inner.get(&new_entry.0);

        if let Some(entry) = entry {
            let result = entry.1.fetch_xor(value, ordering);

            if result ^ value == V::ZERO {
                self.inner.remove(&entry.0);
            }

            result
        } else {
            let result = self.inner.insert(new_entry.clone());

            if let Err(existing) = result {
                // Another thread inserted it first, insert to that value instead.
                return existing.1.fetch_xor(value, ordering);
            }

            V::ZERO
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

        let entry = self.inner.get(&new_entry.0);

        if let Some(entry) = entry {
            let result = entry.1.fetch_max(value, ordering);

            if value >= result && value == V::ZERO {
                self.inner.remove(&entry.0);
            }

            result
        } else {
            if value <= V::ZERO {
                return V::ZERO;
            }

            let result = self.inner.insert(new_entry.clone());

            if let Err(existing) = result {
                // Another thread inserted it first, insert to that value instead.
                return existing.1.fetch_max(value, ordering);
            }

            V::ZERO
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

        let entry = self.inner.get(&new_entry.0);

        if let Some(entry) = entry {
            let result = entry.1.fetch_min(value, ordering);

            if value <= result && value == V::ZERO {
                self.inner.remove(&entry.0);
            }

            result
        } else {
            if value >= V::ZERO {
                return V::ZERO;
            }

            let result = self.inner.insert(new_entry.clone());

            if let Err(existing) = result {
                // Another thread inserted it first, insert to that value instead.
                return existing.1.fetch_min(value, ordering);
            }

            V::ZERO
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

    #[test]
    fn test_hypercounter_concurrency() {
        let counter: Arc<HyperCounter<String, AtomicUsize>> = Arc::new(HyperCounter::new());

        let mut handles = Vec::new();

        for _ in 0..10 {
            let counter = Arc::clone(&counter);

            let handle = std::thread::spawn(move || {
                let key = "key".to_string();

                for i in 0..1_000_000 {
                    if i % 2 == 0 {
                        counter.fetch_add(key.clone(), 1, Ordering::Relaxed)
                    } else {
                        counter.fetch_sub(key.clone(), 1, Ordering::Relaxed)
                    };
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
