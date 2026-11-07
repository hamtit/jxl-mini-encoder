//! A minimal scoped thread pool.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `(0..n).map(f).collect()`, spread over up to `threads` threads.
pub fn par_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = threads.clamp(1, n.max(1));
    if threads == 1 {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<T>>> = Mutex::new((0..n).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let v = f(i);
                    results.lock().unwrap()[i] = Some(v);
                }
            });
        }
    });
    results.into_inner().unwrap().into_iter().map(|v| v.expect("missing result")).collect()
}

/// Run `f` on every task, spread over up to `threads` threads. Tasks are
/// handed out in order, so they should be roughly equal in cost. Used to
/// process disjoint `chunks_mut` bands of an output buffer in parallel.
pub fn par_tasks<T: Send>(tasks: Vec<T>, threads: usize, f: impl Fn(T) + Sync) {
    let threads = threads.clamp(1, tasks.len().max(1));
    if threads == 1 {
        tasks.into_iter().for_each(f);
        return;
    }
    let queue = Mutex::new(tasks.into_iter());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let task = queue.lock().unwrap().next();
                    match task {
                        Some(t) => f(t),
                        None => break,
                    }
                }
            });
        }
    });
}
