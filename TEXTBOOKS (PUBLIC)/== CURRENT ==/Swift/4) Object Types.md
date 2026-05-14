
## Object Type Declarations and Features

- Object types are declared with
	- Enum
	- Struct
	- Class

- Visibility
	- Top level
	- Inside type declaration
	- Function body
- Declarations may contains
	- Initializers
	- Properties
	- Methods
	- Subscripts
	- Object type declarations


### Initializers

- An initializer is a function for producing an instance of an object type

```swift
class Dog {

}

Dog()
```

- Object types may have implicit initializers

#### How to write an initializer

- Keyword `init` with a parameter list, followed by curly braces
- Function that does not involved `func` or a return type
- An object type can have multiple initializers

```swift
class Dog {
	var name = ""
	var license = 0
	init(name:String){
		self.name = name
	}
	init(license:Int){
		self.license = license
	}	
	init(name:String, license:Int){
		self.name = name
		self.license = license
	}
}

let fido = Dog(name:"Fido")
let rover = Dog(license:1234)
let spot = Dog(name:"Spot", license:1357)
```

- Initializer is a function, and a function's parameters can have default values

```swift
class Dog {
	var name = ""
	var license = 0
	init(name:String = "", license:Int = 0){
		self.name = name
		self.license = license
	}
}

let fido = Dog(name:"Fido")
let rover = Dog(license:1234)
let spot = Dog(name:"Spot", license:1357)
let puff = Dog()
```

- Eliminate the assignment of default values
- Must initializer all stored properties
- Counts the initializers with `let`


#### Deferred Initialization of Properties



## Enums
## Structs
## Classes
## Polymorphism

## Casting
## Type References
## Protocols
## Generics
## Extensions

## Umbrella Types

## Collection Types