
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
- Variables have type
	- Variable type can not changed once declared
	- Can be replaced with a difference type
- Type names start with a capital letter
- Variable names start with a small letter

## Functions

- Executable code must remain inside the body of a function
- Delimited by curly braces

```swift
func go() {
	let one = 1
	var two = 2
	two = one
}
```

`main.swift`
- Is the code that is ran when the program starts

## Structure of a Swift File

- Module `import` statements
- Variable declarations
- Function declarations
- Object type declarations

- Only a function body can contain executable code
- Executable code cannot go directly inside a `class` declaration

## Scope and Lifetime

- Scope
	- Things can see things at their own level and at a higher level containing them
- A module
- A file
- Curly braces

- Lifetime
	- A thing lives as long as its surrounding scope lives
	- A global variable lives as long as the file runs
	- A variable at the top of a function or class exists only as long as the instance

## Object Members

```swift
class Manny {
	let name = "manny"
	func sayName() {
		print(name)
	}
}
```

- `name`
	- Variable declared at the top level of an object declaration
	- Property of that object
- `sayName`
	- Function declared at the top level of an object declaration
	- Method of that objects
- Items declared at the top level of an object declaration are called the members of that object
	- Properties
	- Methods
	- Objects
- Members defined the messages that are sent to that object


## Namespaces

- A namespace is a named region of a program
- Namespaces help explain the significance of declaring an object at the top level of an object

```swift
class Manny {
	class Klass {}
}
```

- Code outside Manny, has to specify the namespace explicitly in order to pass through the barrier
- `Manny.Klass`
- Message sending allows you to see its scopes

## Modules

- Top level namespaces are modules
- Swift is a module
- Own app module overshadows any module imports

## Instances

- Object types can be instantiated
	- Class, struct, enum
- Instances can be created by using the object type's name as a function name and calling the function
	- Parentheses
- Send instance messages

```swift
class Dog {
	func bark() {
		print("woof")
	}
}

let fido = Dog()
fido.bark()
```

- Properties and methods are instance properties and methods
- Declare a function without instantiation with a class function or a static function

```swift
class Dog {
	var name = ""
}
let fido = Dog()
fido.name = "Fido"
```

- The value of an instance property is defined with respect to a particular instance
- An instance is responsible for the values for the lifetimes of its properties
- An instance has state and is a device for maintaining state

## Keyword Self

- An instance is an object, and an object is the recipient of messages
- `self`
	- Keyword used whenever an instance of the appropriate type is expected

```swift
class Dog {
	var name = ""
	var saying = "woof"
	func bark() {
		print(self.saying)
	}
	func speak() {
		self.bark()
	}
}
```

- `self` appears only in instance methods
- Refers to this instance

## Privacy

- Change `var` to `let` for constant instance properties
- `private` keyword
	- Hide the property values from other types of objects
- Object members are public by default
- A class declaration defines a namespace
	- Requires that objects use dot notation to refer to functions in the namespace
- Force a reserved word to be an identifier with backticks

```swift
class `func` {
}
```

## Design

- Instance will persist according to the lifetime of the variable
- Instance is visible to other instances according to scope of the variable
- 