
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
	func bark(){
		print(self.whatDogsSay)
	}
	func speak(){
		self.bark()
		print("I'm \(self.name)")
	}
}

// omit self
func speak() {
    bark()
    print("I'm \(name)")
}

// static/class method is accssed through the type
struct Greeting {
	static let friendly = "hello"
	static func beFriendly(){
		print(self.friendly)
	}
}
```
### Subscripts

- A method that is called by appending square brackets containing arguments directly to a reference
- Elements that are accessed by key or by index number

```swift
struct Digit {
	var number : Int
	init(_ n: Int){
		self.number = n
	}
	subscript(ix:Int) -> Int {
		get {
			let s = String(self.number)
			return Int(String(s[s.index(s.startIndex, offsetBy:ix)]))!
		}
	}
}

let d = Digit(1234)
let aDigit = d[1]

struct Digit {
    var number : Int
    init(_ n:Int) {
        self.number = n
    }
    subscript(ix:Int) -> Int {
        get {
            let s = String(self.number)
            return Int(String(s[s.index(s.startIndex, offsetBy:ix)]))!
        }
        set {
            var s = String(self.number)
            let i = s.index(s.startIndex, offsetBy:ix)
            s.replaceSubrange(i...i, with: String(newValue))
            self.number = Int(s)!
        }
    }
}

var d = Digit(1234)
d[0] = 2 // now d.number is 2234

subscript(ix:Int = 0) -> Int {
```

- Parameter names are not externalized
	- Add external name, `subscript(ix ix:Int)`

- An object type can declare multiple subscript methods
- 
### Nested Object Types

- An object type may be declared inside an object type declaration
- A nested type can't refer directly to the surrounding type's instance members
	- Can refer to the surrounding type's static/class members

```swift
class Dog {
	struct Noise {
		static var noise = "woof"
	}
	func bark() {
		print(Dog.Noise.noise)
	}
}

Dog.Noise.noise = "arf"

class Dog {
	static let sound = "ruff"
	struct Noise {
		statuc var noise = "woof"
		func barkTheDog() { bark() } // compile eror
		var othernoise = sound
	}
	func bark() {
		print(Dog.Noise.noise)
	}
}
```

- Code inside `Noise` cannot refer directly to Dog's `bark` method
- Code inside `noise` can refer to `sound` static property

## Enums

- Enum
	- An object type whose instance represent distinct predefined alternative values
	- Express a set of consents that are alternatives to one another
	- Each case is the name of one of the alternatives

```swift
enum Filter {
	case albums
	case playlists
	case podcasts
	case books
}
```

- Write an initializer for an enum
- Instances of an enum with the same case are regarded as equal

```python
let type = Filter.albums

let type : Filter = .albums

func filterExpecter(_ type:Filter) {}
filterExpecter(.albums)

let v = UIView()
v.contentMode = .center

func filterExpecter(_ type:Filter) {
	if type == .albums {
		print("it is album")
	}
}
filterExpecter(.albums)
```

### Raw Values

- Adds a type dec
### 
### 
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