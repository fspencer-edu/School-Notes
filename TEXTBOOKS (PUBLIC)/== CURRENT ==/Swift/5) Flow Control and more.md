## Flow Control

- A computer program has a path of execution
	- Branching
	- Looping
- Chunks are referred to as blocks
	- A condition does not have to be wrapped in parentheses
	- Curly braces can never be omitted in swift

### Branching

- If construct
- Switch statement

#### If construct

- `if`
```swift
if condition {
    statements
}

if condition {
    statements
} else {
    statements
}
```
#### Conditional branching

- `if` followed by a variable declaration and assignment

```swift
if let variable = value {
    // the block
}

if let prog = n.userInfo?["progress"] as? Double {
    self.progress = prog
}

if let ui = n.userInfo, let prog = ui["progress"] as? Double {
    self.progress = prog
}
```
- Conditionally unwrapping an Optional
- Condition list
	- Avoid indentation

#### Switch statement

- A switch statement is a neater way of writing `if...else`
- Contains hidden traps
- Cases are compared against values called tags
- Performed in order
- `default`
- Exhaustive

```swift
switch tag {
case pattern1:
    statements
case pattern2:
    statements
default:
    statements
}

switch i {
case 1:
    print("You have 1 thingy!")
case 2:
    print("You have 2 thingies!")
default:
    print("You have \(i) thingies!")
}
```
- Underscore `(_)` to absorb all values without using them
- Declaration of a local variable to absorb all values and use value
- `contains` or range pattern
- Optional wrapping

```swift
switch i {
case 1:
    print("You have 1 thingy!")
case _:
    print("You have many thingies!")
}

switch i {
case 1:
    print("You have 1 thingy!")
case let n:
    print("You have \(n) thingies!")
}

switch i {
case 1:
    print("You have 1 thingy!")
case 2...10:
    print("You have \(i) thingies!")
default:
    print("You have more thingies than I can count!")
}

switch i {
case 1?:
    print("You have 1 thingy!")
case let n?:
    print("You have \(n) thingies!")
case nil: break
}
```

#### If case

- Use the same sort of pattern syntax as a switch statement
- Followed by an equal sign and a tag

```swift
if case let .number(n) = err {
    print("The error number is \(n)")
}

if case let .number(n) = err, n < 0 {
    print("The negative error number is \(n)")
}
```
#### Conditional evaluation

- Use a define-and-call anonymous function
	- Write a variable while being a branching construct

```swift
let title : String = {
    switch type {
    case .albums:
        return "Albums"
    case .playlists:
        return "Playlists"
    case .podcasts:
        return "Podcasts"
    case .books:
        return "Books"
    }
}()
```
- Two-pronged condition
	- Ternary operator `(?:)`
- Use `??` to chain expressions

```swift
condition ? exp1 : exp2
let someNumber = i1 as? Int ?? i2 as? Int ?? 0

```
### Loops

- Repeat a block of code with some simple difference on each iteration
#### While loops

```swift
while condition {
    statements
}

repeat {
    statements
} while condition
```
#### For loops

```swift
for variable in sequence {
    statements
}

var iterator = (1...5).makeIterator()
while let i = iterator.next() {
    print(i) // 1, 2, 3, 4, 5
}
```

- Enumerate through a sequence
- `lazy` sequence
	- Short-circuiting the loop
- Trailing closures
	- Parentheses are needed

```swift
for i in arr.map ({$0*2}) {
    print(i)
}
```
### Jumping

- Interrupt code's progress completely and jump to a different place

#### Return

- Return statement
	- One function calls another
#### Short-circuiting and labels

- `fallthough`
	- A switch case aborts execution of the current case code, and beings executing the next case
- `continue`
	- Aborts execution of current and proceeds to next iteration
- `break`
	- Aborts the current construct and proceeds after the end of the construct

#### Throwing and catching errors

- 
#### Defer Statement
#### Aborting the whole program
#### Guard
 
## Privacy

### Private and Fileprivate

### Public and Open

### Privacy Rules
 
## Introspection

## Operators
## Memory Management

### Memory Management of Reference Types

#### Weak references
#### Unowned references
#### Stored anonymous functions

## Miscellaneous Swift Language Features

### Synthesized Protocol Implementations

#### Equatable
#### Hashable
#### Comparable
### Key Paths

### Instance as Function

### Dynamic Membership

### Property Wrappers
### Custom String Interpolation
### Reverse Generics
### Result Builders
### Result


