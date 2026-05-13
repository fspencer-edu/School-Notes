
- A complete Swift command is a statement
- Swift text file consists of multiple lines of text
- Line breaks are meaningful
- Semi-colon is optional for line breaks

```swift
print("hello")

// comments
```

- Constructs use curly braces as delimiters

```swift
class Dog {
	func bark() {
		print("woof")
	}
}
```

- Swift is a complied language
	- Must be complied to a lower-level language before running

## Objects

- Everything is an object
- Something you can send a message to
	- Imperative instruction
- Message-sending is dot notation

```swift
object.message()
```

- Noun = object
- Verb = message

- An object type can be extended in Swift
- Define a custom message on an object type
- In Swift there are no scalar
- All types are object types

### 3 types of Object Types

- An object is a class or an instance of a class
- `1` is a struct, Int
	- `1` is an instance of a struct

1) Classes
2) Structs
3) Enums
4) Actors

## Variables

- A variable is a name for an object
	- Object reference
- All variables must be declared
	- `let`
	- `var`
- Declaration is followed by initialization
	- Assignment operator
	- Does not assert equality
- `let`
	- Used for constants
	- Cannot have its initial value replaced
	- More efficient
	- 