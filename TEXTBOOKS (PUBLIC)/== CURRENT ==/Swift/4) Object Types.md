
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

- All instance properties must be initialized either in declaration or through an initializer
	- Assign a default value
	- Declare an instance property as a `var` having an Optional type

```python
@IBOutlet var myButton: UIButton!

var albums : [MPMediaItemCollection]?
```

#### Referring to self

```python
struct Cat {
    var name : String
    var license : Int
    init(name:String, license:Int) {
        self.name = name
        meow() // too soon - compile error
        self.license = license
    }
    func meow() {
        print("meow")
    }
}
```

- To call `meow` is implicitly a reference to `self.meow()`
- Needs to move after `name` and `license` are initialized

#### Delegating initializers

- Initializers within an object type cal call one another using `self.init()`
- Delegating Initializer
	- Initializer that calls another Initializer

```python
struct Digit {
	var number : Int
	var meaningOfLife : Bool
	init(number:Int){
		self.number = number
		self.meaningOfLife = false
	}
	init() { # delegating init
		self.init(number:42)
		self.meaningOfLife = true
	}
}
```

- A delegating initializer cannot set a constant property (`let`)

#### Failable initializers

- An initializer can return an Optional wrapping the new instance
- `nil` is returned to signal failure

```python
class Dog {
	let name : String
	init?(name:String){
		if name.isEmpty{
			return nil
		}
		self.name = name
	}
}

# exit early
class Dog {
    let name : String
    init?(name:String) {
        if !name.isEmpty {
            self.name = name
            return
        }
        return nil
    }
}
```

### Properties

- Property
	- Variable
	- Declared at the top level of an object type declaration
	- Fixed type
	- `var` or `let`
	- An instance property can also be declared `lazy`
- A stored instance property must have an initial value

#### How properties are accessed

- A property is an instance property
- Accessed only through an instance

```python
class Dog {
	let name : String
	let license : Int
	init(name:String, license:Int){
		self.name = name
		self.license = license
	}
}

let fido = Dog(name:"Fido", license:1234)
let spot = Dog(name:"Spot", license:1293)
let fidoName = fido.name
```

- Access the `name` property through the instance

- Static/class property
	- Access through the type
	- Scoped to the type

```python
struct Greeting {
	static let friendly = "hello"
	static let leaving = "good bye"
}
```
#### Property initialization and self

- Property declaration that assign an initial value to the property cannot fetch an instance property or call an instance method
- Make this a computed property
	- Refers to `self` in a getter or setter function

```python
class Moi {
    let first = "Matt"
    let last = "Neuburg"
    let whole = self.first + " " + self.last // compile error
}

# computed property
class Moi {
    let first = "Matt"
    let last = "Neuburg"
    var whole : String {
        self.first + " " + self.last
    }
}

# lazy
class Moi {
    let first = "Matt"
    let last = "Neuburg"
    lazy var whole = self.first + " " + self.last
}

class Moi {
    let first = "Matt"
    let last = "Neuburg"
    lazy var whole : String = {
        var s = self.first
        s.append(" ")
        s.append(self.last)
        return s
    }()
}
```
- Static property can be initialized with reference to another
	- Lazy

```python
struct Greeting {
    static let friendly = "hello there"
    static let hostile = "go away"
    static let ambivalent = friendly + " but " + hostile
}
```

### Methods

- Method
	- A function that is declare at the top level of an object declaration
	- Instance method
	- Accessed only through an instance

```swift
class Dog {
	let name : String
	let license: Int
	let whatDogSay = "woof"
	init(name:String, license:Int){
		self.name = name
		self.license = license
	}
	func bar
}
```

#### 
#### 
#### 
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