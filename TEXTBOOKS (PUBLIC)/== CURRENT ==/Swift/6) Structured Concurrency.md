
- Concurrency
	- Ability of a computer program to do more than one thing at the same time
- Explicit concurrency
	- Communication from background code
	- Running code in the background

## Multithreading

- Concurrency is traditionally envisioned in terms of threads
- Threads are the low-level expression of simultaneous subprocess execution

### The Main Thread

- Only one main thread
- Blocking
	- A thread that prevents any other code from running on that thread

### Background Threads

- Shared data
- Lock
	- Helps prevent race condition and dead clocks

### Asynchronous Code

- Code that might be called at some unknown future time

#### What asynchronous code looks like

- A typical asynchronous code architecture is a completion handler

```swift
dataTask(
    with: URL,
    completionHandler: @escaping (Data?, URLResponse?, Error?) -> Void)
    -> URLSessionDataTask
```

#### Returning a value

- Code containing a time inversion is hard to reason
#### Throwing an error

#### Summary

- Issues of asynchronous code
	- Confusion order
	- Can't return a value
	- Can't throw an error
## Structured Concurrency Syntax

- Asynchronous cod is marked
- Asynchronous code runs in order
- Asynchronous code can return a value
- Asynchronous code can throw an error

### async/await

- `async`
	- Masks an asynchronous method
- `await`
	- Used to call an asynchronous method
### async Contexts

## Tasks

- A Task object represents the notion of a task
- Basis of all asynchronous activity
- Tasks are also the atoms of structured concurrency
	- `priority:`
	- `operation:`

```swift
override func viewDidLoad() {
    super.viewDidLoad()
    let url = URL(string: "https://www.apeth.com/pep/manny.jpg")!
    Task {
        do {
            let data = try await self.download(url: url)
            print(data)
        } catch {
            print(error)
        }
    }
    print(url)
}
```

## Wrapping a Completion Handler
## Multiple Concurrent Tasks

## Asynchronous Sequences
## Actors
## Context Switching
## More About Tasks
## More About Actors

## Sendable