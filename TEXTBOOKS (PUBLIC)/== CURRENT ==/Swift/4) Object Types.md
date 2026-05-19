
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

- Adds a type declaration to an enum
- Every case then carries with it a fixed value of that type
- Types attached to an enum's cases in this way are limited to numbers and strings
- Values assigned must be literals

```swift
enum PepBoy : Int {
	case manny
	case moe
	case jack
}
```

- `.manny` carries a value of `0`, `.moe` of 1

```swift
enum Filter : String {
	case albums
	case playlists
	case podcasts
	case books
}
```

- The values carried by the cases are called their raw values
- An enum with a type declaration implicitly adopts the RawRepresentable protocol
	- Implicitly has an `init(rawValue:)` initializer and a `rawValue` property

```swift
let type = Filter.albums
print(type.rawValue)
let type = Filter(rawValue:"Albums")

// optional wrapping
let type = Filter(rawValue:"Albums")
if type == .albums
```

- Raw value associated with each case must be unique within the enum

```swift
struct Dog {
    let name: String
}
func dogExpecter(_ dog: Dog) {
    print(dog.name)
}

dogExpecter(.init(name:"Fido"))
```
### Associated Values

- Raw values are fixed in the enum's declaration
- Construct a case whose constant value can be set when the instance is created
	- Associated value
- Append the value to the name of the case an expression

```swift
enum MyError {
	case number(Int)
	case message(String)
	case fatal
}

let err : MyError = .number(4)
```

- Optional is an enum with two cases, `.none` and `.some`
- Labels must be initialized

```swift
enum MyError2 {
    case number(Int)
    case message(String)
    case fatal(n:Int, s:String)
}
let err : MyError2 = .fatal(n:-12, s:"Oh the horror")
```

- Cannot use `==` operation, instead `Equatable` protocol
### Enum Case Iteration

- Define a list as a static property of the enum
- The list of cases can be generated automatically
- CaseIterable protocol

```swift
enum Filter : String {
    case albums = "Albums"
    case playlists = "Playlists"
    case podcasts = "Podcasts"
    case books = "Audiobooks"
    static let cases : [Filter] = [.albums, .playlists, .podcasts, .books]
}

enum Filter : String, CaseIterable {
    case albums = "Albums"
    case playlists = "Playlists"
    case podcasts = "Podcasts"
    case books = "Audiobooks"
    // static allCases is now [.albums, .playlists, .podcasts, .books]
}
```
### Enum Initializers

- An explicit enum initializer must do what default initialization does
- Return a case of this enum

```swift
enum Filter : String, CaseIterable {
    case albums = "Albums"
    case playlists = "Playlists"
    case podcasts = "Podcasts"
    case books = "Audiobooks"
    init(_ ix:Int) {
        self = Filter.allCases[ix]
    }
}

let type1 = Filter.albums
let type2 = Filter(rawValue:"Playlists")!
let type3 = Filter(2) // .podcasts

// failable
enum Filter : String, CaseIterable {
    case albums = "Albums"
    case playlists = "Playlists"
    case podcasts = "Podcasts"
    case books = "Audiobooks"
    init?(_ ix:Int) {
        if !Filter.allCases.indices.contains(ix) {
            return nil
        }
        self = Filter.allCases[ix]
    }
}
```
### Enum Properties

- An enum can have instance and static properties
- An enum instance property can't be a stored property

### Enum Methods

- An enum can have instance and static methods

```python
enum Shape {
    case rectangle
    case ellipse
    case diamond
    func addShape (to p: CGMutablePath, in r: CGRect) -> () {
        switch self {
        case .rectangle:
            p.addRect(r)
        case .ellipse:
            p.addEllipse(in:r)
        case .diamond:
            p.move(to: CGPoint(x:r.minX, y:r.midY))
            p.addLine(to: CGPoint(x: r.midX, y: r.minY))
            p.addLine(to: CGPoint(x: r.maxX, y: r.midY))
            p.addLine(to: CGPoint(x: r.midX, y: r.maxY))
            p.closeSubpath()
        }
    }
}

enum Filter : String, CaseIterable {
    case albums = "Albums"
    case playlists = "Playlists"
    case podcasts = "Podcasts"
    case books = "Audiobooks"
    static subscript(ix: Int) -> Filter {
        Filter.allCases[ix] // warning, no range checking
    }
}
```

- An enum instance method that modifies the enum is marked as `muting`
- `advance`
	- Transform a Filter instance into an instance of the next case in the sequence

```swift
enum Filter : String, CaseIterable {
    case albums = "Albums"
    case playlists = "Playlists"
    case podcasts = "Podcasts"
    case books = "Audiobooks"
    mutating func advance() {
        let cases = Filter.allCases
        var ix = cases.firstIndex(of:self)!
        ix = (ix + 1) % cases.count
        self = cases[ix]
    }
}

var type = Filter.books
type.advance() // type is now Filter.albums
```

### Enums?

- An enum is a switch whose states have names

## Structs

- Struct is the Swift object type par excellence

### Struct Initializers

- A struct does not have en explicit initializers
- No stored properties
- If an explicit initializer is added, it losses the implicitly initializer

```swift
struct Digit {
	var number = 32
}

struct Digit {
	var number = 42
	init(number:Int) {
		self.number = number
	}
}
```
- Store properties that do not have an explicit initializer automatically gets an implicit initializer derived from its instance properties
	- Memberwise initializer

```swift
struct Test {
	var numbers = 42
	var name : String
	let age : Int
	let greeting = "Hello"
}

et t1 = Test(number: 42, name: "matt", age: 67)
let t2 = Test(name: "matt", age: 67)
```

### Struct Properties

- A struct can have instance properties and static properties

```python
struct Digit {
	var number : Int
	init(_ n:Int) {
		self.number = n
	}
}

var d = Digit(123)
d.number = 42
```
### Struct Methods

- Struct can have instance and static methods

```swift
struct Digit {
	private var number : Int
	init(_ n:Int) {
		self.number = n
	}
	mutating func changeNumberTo(_ n:Int){
		self.number = n
	}
}

var d = Digit(123)
d.changeNumberTo(42)
```

- Degenerate struct
	- Consists of entirely static members

```swift
enum Default {
    static let rows = "CardMatrixRows"
    static let columns = "CardMatrixColumns"
    static let hazyStripy = "HazyStripy"
}
```

## Classes

- A class is similar to a struct
	- Reference type
		- Mutability
		- Multiple references
	- Inheritance

### Value Types and Reference Types

- Enums and structs are value types
- Classes are reference types

#### Class instances are mutable

- A value type is not mutable in place

```swift
struct Digit {
	var number : Int
	init(_ n:Int) {
		self.number = n
	}
}
```

- Impossible to mutable a value type instance if the reference is declared with `let`
- Classes are not value types
	- Mutable in place

```swift
class Dog {
	var name : String = "Fido"
}

let rover = Dog()
rover.name = "Rover"
```

#### Class instance references are pointers

- Reference holds a pointer to the instance
- There can be multiple references to the same object
	- Not true of structs and enums
#### Advantages of value types vs. references types

- Class instances are more complicated behind the scenes
	- More overhead
- Prefer a value type over a reference type
- Classes are efficient to pass around
	- Independent readily
	- Recursive references

```swift
struct Dog {
	var puppy: Dog?
}

enum Node {
    case none(Int)
    indirect case left(Int, Node)
    indirect case right(Int, Node)
    indirect case both(Int, Node, Node)
}
```
- `indirect`
	- Enum's case associated value can be an instance of that enum
### Subclass and Superclass

- Two classes can be subclass and superclass of one another
- Cocoa
	- One base class
	- NSObject
	- All other classes are subclasses, at some level
	- Tree hierarchy

#### Inheritance

- Share functionality

```swift
class Quadruped {
	func walk() {
		print("walk walk walk")
	}
}
class Dog : Quadruped {}
class Cat : Quadruped {}

let fido = Dog()
fido.walk()
```
- A class declaration can prevent the class from being subclassed by preceding the class declaration with `final`

#### Additional functionality

- Subclass consists of the methods inherited from superclass, and subclass specific methods

```swift
class Quadruped {
    func walk () {
        print("walk walk walk")
    }
}
class Dog : Quadruped {
    func bark () {
        print("woof")
    }
    func barkAndWalk() {
        self.bark()
        self.walk()
    }
}

let fido = Dog()
fido.barkAndWalk() // woof walk walk walk
```
#### Overriding

- Subclass can redefine a method inherited from its superclass

```swift
class Quadruped {
	func walk() {
		print("walk")
	}
}
class Dog : Quadruped {
	func bark() {
		print("woof:)
	}
}
class NoisyDog : Dog {
	override funk bark() {
		print("woof woof")
	}
}
```

- An override exists only when the subclass redefined the same method that is inherits from a superclass
	- Same name and external parameters
- A method override can be replaced with an Optional wrapping the superclass

```swift
class Dog {
	func barkAt(cat:Kitten) {}
}
class NoisyDog : Dog {
	override func barkAt(cat:Cat) {}
}
```

 - A class declaration can prevent from being overridden by a subclass with `final`

#### The keyword super

- `super`
	- Override something in the subclass, as well has override the superclass

```swift
class Dog : Quadruped {
	func bark() {
		print("woof")
	}
}
class NoisyDog : Dog {
	override func bark() {
		for _ in 1...3 {
			super.bark()
		}
	}
}
```

### Class Initializers

- Initialization of a class instance is more complicated than initialization of a struct or enum instance
- All properties should have an initial value
- When a subclass is initialized, its superclass properties should be initialized
#### Kinds of class initializer

- Designated initializer
	- A class can be instantiated only through a class to one of its designated initializers
	- Does not delegate to another initializer in the same class
- Convenience initializer
	- Marked with `convenience`
	- Facade for a designated initializer
	- Must contain `self.init()`

```swift
// designated initializer
class Dog {
}
let d = Dog()

class Dog {
	var name = "Fide"
}
let d = Dog()

class Dog {
    var name = "Fido"
    init(name:String) {self.name = name}
}
let d = Dog(name:"Rover") // ok
let d2 = Dog() // compile error
```

- A designated initializer cannot, except in order to initialize a property say `self`
- A convenience initializer is a delegating initializer

#### Subclass initializers

- No declared initializers
- Convenience initializers only
- Designated initializers
- Designated and convenience initializers
- Override initializers
- Failable initializers

#### Subclass initializer examples

```swift
class Dog {
	var name : String
	var license : Int
	init(name:String, license:Int) {
		self.name = name
		self.license = license
	}
	convenience init(license:Int) {
		self.init(name:"Fido", license:license)
	}
}
class NoisyDog : Dog{
}

let nd1 = NoisyDog(name:"Fido", license:1)
let nd2 = NoisyDog(license:2)
let nd3 = NoisyDog() // compile error
```

#### Required Initializers

- A class initializer may be preceded by `required`
- A subclass may not lack this initializer
- Must override this initializer and mark the override `required`

```swift
class Dog {
    var name : String
    required init(name:String) {
        self.name = name
    }
}
class NoisyDog : Dog {
    var obedient = false
    init(obedient:Bool) {
        self.obedient = obedient
        super.init(name:"Fido")
    }
    required init(name:String) {
        super.init(name:name)
    }
}
```

### Class Deinitializer

- A function declared with the keyword `deinit`
- Called by the runtime when an instance of this class goes out of existence
- Subclass deinitializer is called before the superclass's
- Class feature only
- Perform internal cleanup
- Log to the console

### Class Properties

- A subclass can override its inherited properties
	- Same name and type, marked with `override`
	- Cannot be a stored property

### Static/Class Members
#### Static methods vs. class methods

- Static
	- Cannot be overridden
- Class
	- Can be overridden in a subclass

```swift
class Dog {
	static func whatDogsSay() -> String {
		 return "woof"
	}
	func bark() {
		print(Dog.whatDogsSay())
	}
}
```
#### Static vs. Class Properties

- A static property can be stored
- A class property must be a computer property

```swift
class Dog {
	static var whatDogsSay = "woof"
	func bark() {
		print(Dog.whatDogsSay)
	}
}

// computer property
class Dog {
    class var whatDogsSay : String {
        return "woof"
    }
    func bark() {
        print(Dog.whatDogsSay)
    }
}
```
## Polymorphism

- Substitution
- Internal identity

```swift
class Dog {
}
class NoisyDog : Dog {
}
let d : Dog = NoisyDog()

class Dog {
    func bark() {
        print("woof")
    }
}
class NoisyDog : Dog {
    override func bark() {
        for _ in 1...3 {
            super.bark()
        }
    }
}

func tellToBark(_ d:Dog) {
    d.bark()
}
var nd = NoisyDog()
tellToBark(nd) // what will happen??????
```
- Pass `nd`, typed a NoisyDog, where a Dog is expected
- The internal identity rule
- `self` depends upon the type of the actual instance

```swift
class Dog {
    func bark() {
        print("woof")
    }
    func speak() {
        self.bark()
    }
}
class NoisyDog : Dog {
    override func bark() {
        for _ in 1...3 {
            super.bark()
        }
    }
}

let nd = NoisyDog()
nd.speak() // woof woof woof
```

- Polymorphism applies to Optional types in the same way that is applies to the type of thing wrapped by the Option

```swift
var d : Dog?
d = Dog()
d = NoisyDog()
d = Optional(NoisyDog())
```

- Optional are covariant
- UIViewController is a subclass, and override methods
- Polymorphism requires dynamic dispatch
	- Compiler can't perform certain optimizations
	- Runtime has to think about a message to a class instance
	- Reduce dynamic dispatch by declaring a class or a class member `final` or `private`

## Casting

- The messages that the compiles will permit to be sent to an object reference depend upon the reference's declared type
- Internal identity principle of polymorphism changes an objects real tupe from reference's declared type

```swift
class Dog {
	func bark() {
		print("woof")
	}
}
class NoisyDog : Dog {
	override func bark() {
		super.bark(); super.bark()
	}
	func beQuiet() {
		self.bark
	}
}

func tellToHush(_ d:Dog) {
    d.beQuiet() // compile error
}
let nd = NoisyDog()
tellToHush(nd)
```

- Cannot send `beQuiet` message to the reference `d` inside the function body, since it is typed as Dog
- Casting
	- Us `as` to claim a type

### Casting Down

- You cannot cast a String to an Int
- Cast a superclass to a subclass
	- `as~`
	- Forcing the compiler to do something

```swift
func tellToHush(_ d:Dog) {
	(d as! NoisyDog).beQuiet()
}
let nd = NoisyDog()
tellToHush(nd)
```

### Type Testing and Casting Down Safely

- Casting down is a way of telling the compiler to relax its strict type checking
- Test the type of an instance at runtime
	- `is`
- `as?`
	- Cast down, with the option of failure

```swift
func tellToHush(_ d:Dog) {
    let d = d as! NoisyDog // crash
    d.beQuiet()
}
let d = Dog()
tellToHush(d)

func tellToHush(_ d:Dog) {
    if d is NoisyDog {
        let d = d as! NoisyDog
        d.beQuiet()
    }
}

func tellToHush(_ d:Dog) {
    let d = d as? NoisyDog // an Optional wrapping a NoisyDog
    if d != nil {
        d!.beQuiet()
    }
}

func tellToHush(_ d:Dog) {
    (d as? NoisyDog)?.beQuiet()
}
```
### Bridging the Objective-C

- Interchange between Swift and Objective-C
- Cast a Swift String to a Cocoa NSString

```swift
let s : NSString = "howdy"
let s2 = "howdy"
let s3 : NSString = s2 as NSString
let i : NSNumber = 1 as NSNumber

let name = "MyNib" // Swift String
let vc = ViewController(nibName:name, bundle:nil)
```

## Type References

### From Instance to Type

- `type(of:)`

```swift
let d : Dog = NoisyDog()
print(type(of:d))
```
### From self to Type

- An instance should be able to refer to its own type
- Use `Self` to refer to the current type

```swift
class Dog {
	class var whatDogsSay : String {
		return "woof"
	}
	func bark() {
		print(Dog.whatDogsSay)
	}
}

class Dog {
    class var whatDogsSay : String {
        return "woof"
    }
    func bark() {
        print(Self.whatDogsSay) // woof
    }
}
```

- Factory method
	- Write an instance method version
	- Creating objects in a superclass while allowing subclasses to determine the specific concrete class to instantiate

### Type as Value

- Treat an object type as a value
- Metatype
	- Declare that an object type is acceptable with 1Type
	- Use an object type as a value

```swift
func dogTypeExpecter(_ whattype:Dog.Type) {
}

dogTypeExpecter(Dog.self)
let d = Dog()
dogTypeExpecter(type(of:d))
```

### Summary of Type Terminology

- `type(of:)`
- `Self`
- `.Type`
- `.self`
### Comparing Types

- Type reference can be compared to one another
	- `==`
	- `is`
	- `.Type`

```swift
func dogTypeExpecter(_ whattype:Dog.Type) {
    let equality = whattype == Dog.self
    let typology = whattype is Dog.Type
}
```
## Protocols

- Protocols
	- Way of expressing commonality between otherwise unrelated types
	- A object type
	- No protocol objects
	- Lightweight list of properties and methods
	- No values, methods have no code
- Adopting the protocol
	- Object type can declare that is belongs to a protocol type
- Conforming to the protocol
	- Protocol promises to implement the properties and methods listed by the protocol
- A protocol can also declare a method and provide its implementation
	- Protocol extensions

```swift
protocol Flier {
	func fly()
}

struct Bird : Flier {
	func fly() {
	}
}
```
### Why Protocols?

- A protocol is a kind of type
- Use Flier as a type when declaring the type of a variable or a function parameter

```swift
func tellToFly(_ f:Flier){
	f.fly()
}
```
- Polymorphism applies
- An object with a `fly` method is not automatically a Flier
- Object type must formally adopt the protocol

```swift
func tellToFly(_ f:Flier) {
	f.fly()
}
struct Bee {
	func fly() {
	}
}
let b = Bee()
tellToFly(b) // compile error
```

- Bee cannot be sent the `fly` message
- Takes Flier parameter

```swift
func tellToFly(_ f:Flier) {
    f.fly()
}
struct Bee : Flier {
    func fly() {
    }
}
let b = Bee()
tellToFly(b)
```
### Adopting a Library Protocol

- CustomStringConvertilbe protocol
	- `description` String property

```swift
enum Filter : String, CustomStringConvertible {
    case albums = "Albums"
    case playlists = "Playlists"
    case podcasts = "Podcasts"
    case books = "Audiobooks"
    var description : String { return self.rawValue }
}

let type = Filter.albums
print("It is \(type)") // It is Albums
print(type) // Albums
let s = String(describing:type) // Albums
```

- Give any object type the power of string conversion
- Type can adopt more than one protocol

```swift
struct MyType : CustomStringConvertible, TextOutputStreamable, Strideable {
    // ...
}
```
### Protocol Type Testing and Casting

- `is`
	- Test object's declared or real type

```swift
func isBird(_ f:Flier) -> Bool {
    return f is Bird
}
```
### Declaring a Protocol

- Can take place only at the top level of a file
- `protocol`
	- Properties
	- Methods
- A protocol can adopt one or more protocols
### Protocol Composition

- Avoid formally declaring the protocol in the first place by specifying the protocol combination
	- Join with `&`
	- Protocol composition

```swift
func f(_ x: CustomStringConvertible & CustomDebugStringConvertible) {
}

// composite of a class type
protocol MyViewProtocol {
    func doSomethingReallyCool()
}
class ViewController: UIViewController {
    var v: (UIView & MyViewProtocol)?
    func test() {
        self.v?.doSomethingReallyCool() // a MyViewProtocol requirement
        self.v?.backgroundColor = .red // a UIView property
    }
}
```
### Class Protocols

```swift
protocol MyViewProtocol : UIView {
    func doSomethingReallyCool()
}
class ViewController: UIViewController {
    var v: MyViewProtocol? // and therefore a UIView
    func test() {
        self.v?.doSomethingReallyCool() // a MyViewProtocol requirement
        self.v?.backgroundColor = .red // a UIView property
    }
}
```

- Resulting type can take advantage of special memory management features that apply only to classes

```swift
protocol SecondViewControllerDelegate : AnyObject {
    func accept(data:Any)
}
class SecondViewController : UIViewController {
    weak var delegate : SecondViewControllerDelegate?
    // ...
}
```
### Optional Protocol Members

- A protocol member can be declared option
- Explicitly exposed to Objective-C
	- `@objc`


```swift
@objc protocol Flier {
	@objc optional var song : String {get}
	@objc optional func sing()
}
```
#### Optional Properties

- Swift solves the problem by wrapping its fetched values in an Optional

```swift
@objc protocol Flier {
	@objc optional var song : String {get}
}
let f : Flier = Bird()
let s = f.song
```
#### Optional Methods

- Unwrapping is not the result of the method call

```swift
@objc protocol Flier {
    @objc optional func sing()
}
let f : Flier = Bird()
f.sing?()
```
### Implicitly Required Initializers

- An initializer declared in a protocol is implicitly required
- A class that adopts this protocol is forced to make that requirement explicit
- Designate initializer as `required`

```swift
protocol Flier {
    init()
}
class Bird : Flier {
    required init() {}
}
```
### Expressible by Literal

- Protocols declared in the Swift header
	- ExpressibleByNilLiteral
	- ExpressibleByBooleanLiteral
	- ExpressibleByIntegerLiteral
	- ExpressibleByFloatLiteral
	- ExpressibleByStringLiteral
	- ExpressibleByExtendedGraphemeClusterLiteral
	- ExpressibleByUnicodeScalarLiteral
	- ExpressibleByArrayLiteral
	- ExpressibleByDictionaryLiteral

- A literal can appear where an instance of your object type is expected

```swift
struct Nest : ExpressibleByIntegerLiteral {
    var eggCount : Int = 0
    init() {}
    init(integerLiteral val: Int) {
        self.eggCount = val
    }
}
```
## Generics

- A generic is a placeholder for a type
- Optional with a specific known type

```swift
enum Optional<Wrapped> : ExpressibleByNilLiteral { 
    case none
    case some(Wrapped) 
    init(_ some: Wrapped) 
    // ...
}
```

- The process of substitution is called resolving or specializing the generic

```swift
let s = Optional("howdy")

let s : Optional<String> = "howdy"
```
### Generic Declarations

- Generic protocol with `Self`
	- Placeholder meaning the type of the adopter

```swift
protocol Flier {
    func flockTogetherWith(_ f:Self)
}
```

- Generic protocol with associated type
	- Turns the protocol into a generic
	- Associated type name is a placeholder

```swift
protocol Flier {
    associatedtype T
    func flockTogetherWith(_ f:T)
    func mateWith(_ f:T)
}
```

- Generic functions

```swift
func takeAndReturnSameThing<T> (_ t:T) -> T {
    print(T.self)
    return t
}
```

- Generic object types

```swift
struct HolderOfTwoSameThings<T> {
    var firstThing : T
    var secondThing : T
    init(thingOne:T, thingTwo:T) {
        self.firstThing = thingOne
        self.secondThing = thingTwo
    }
}
```

### Type Constraints

- A generic declaration can limit the types that are eligible to be used for resolving a particular placeholder
	- Type constraint
	- Colon and type
- Class name
- Protocol name

```swift
protocol Flier {
    func fly()
}
protocol Flocker {
    associatedtype T : Flier // *
    func flockTogetherWith(f:T)
}
struct Bee : Flier {
    func fly() {}
}
struct Bird : Flocker {
    func flockTogetherWith(f:Bee) {}
}
```

- A generic protocol type can be used only as a type constraint

### Explicit Specialization

- Placeholder's type has been resolved through inference
- Explicit specialization
	- Resolve type manually

- Generic protocol with association
	- The adopter of a protocol can resolve an associated type manually through a type alias

```swift
protocol Flier {
    associatedtype T
}
struct Bird : Flier {
    typealias T = String
}
```

- Generic object type
	- User of a generic object type can resolve a placeholder type manually using the same angle bracket syntax used to declared the generic in the first place

```swift
class Dog<T> {
    var name : T?
}
let d = Dog<String>()
```

### Genetic Types and Covariance

- A generic object type can't be used as the declared type of anything

```swift
class Dog<T> {
    func speak(_ what:T) {}
}
var d: Dog? // compile error
```

- Only resolved generic object types can be a declared type

```swift
class Dog<T> {
    func speak(_ what:T) {}
}
var d: Dog<Int>? // that's a type
var d2: Dog<String>? // that's a different type
// ... and so on ...

// inferred
class Node<T> {
    let value:T
    let parent:Node?
    init(_ value:T, parent:Node?) {
        self.value = value
        self.parent = parent
    }
}
```
- A generic type specialized to a subtype is not polymorphic with respect to the same generic type specialized to a supertype
- Covaiant
	- The polymorphic relationship between the specializations of the placeholders is applied to the generic types themselves

### Associated Type Chains

- When a generic placeholder is constrained to a generic protocol with an associated type, refer to the type using dot-notation

```swift
protocol Fighter {
	associatedtype Enemy : Fighter
}

struct Soldier : Fighter {
    typealias Enemy = Archer
}
struct Archer : Fighter {
    typealias Enemy = Soldier
}

struct Camp<T:Fighter> {
}
```
### Where Clauses

- Express a type constraint
	- Generic function
	- Generic type
	- Generic protocol
	- Associated type in a generic protocol

```swift
func flyAndWalk<T> (_ f:T) where T: Flier {}
func flyAndWalk2<T> (_ f:T) where T: Flier & Walker {}
func flyAndWalk2a<T> (_ f:T) where T: Flier, T: Walker {}
```

## Extensions

- An extension is a way of injecting code into an object type that has already been declared elsewhere
	- Extending an existing object type
	- Adds functionality
	- `extension`
	- Cannot declared a stored property
	- Cannot declared a designated initializer
	- Cannot override an existing member

```swift
extension CGRect {
    var center : CGPoint {
        return CGPoint(x:self.midX, y:self.midY)
    }
}

extension UIColor {
    static var myGolden : UIColor {
        return self.init(
            red:1.000, green:0.894, blue:0.541, alpha:0.900
        )
    }
}

```
### Extending Protocols

- Add methods and properties to the protocol
- Methods and properties are not requirements

```swift
protocol Flier {
}
extension Flier {
    func fly() {
        print("flap flap flap")
    }
}
struct Bird : Flier {
}

let b = Bird()
b.fly()
```
### Extending Generics

- The generic's placeholder type name are visible to the extension
- `where` clause
	- Limits which resolutions of the generic placeholder can call the code injected by this extension
- Array is a generic struct whose placeholder type is called Element
- Conditional conformance
	- An extension with a where clause

## Umbrella Types

- Capable of embracing multiple real types under a single heading

### Any

- Universal Swift umbrella type
- Any object of function can be passed, without casting

```swift
func anyExpecter(_ a:Any) {}
anyExpecter("howdy")     // a struct instance
anyExpecter(String.self) // a struct type
anyExpecter(Dog())       // a class instance
anyExpecter(Dog.self)    // a class type
anyExpecter(anyExpecter) // a function
```
- Cast down
	- Any object as a more specific type
	- `anything`

```swift
if anything is String {
    let s = anything as! String
    // ...
}

let ud = UserDefaults.standard
ud.set(Date(), forKey:"now")
```

### AnyObject

- An empty protocol with the special feature that all class types conform to it 
#### Suppressing type checking
#### Object Identity
### AnyClass


## Collection Types

### Array

#### Array casting and type testing

#### Array comparison
#### Arrays are value types
#### Array subscripting
#### Nested arrays
#### Basic array properties and methods

#### Array enumeration and transformation
#### Swift Array and Objective-C NSArray
### Dictionary

#### Dictionaries have no order
#### Dictionary casting and comparison
#### Basic dictionary properties and enumeration
#### Swift Dictionary and Objective-C NSDictionary
### Set
#### Option sets
#### Swift Set and Objective-C NSSet
### OrderedSet and OrderedDictionary
#### OrderedSet
#### OrderedDictionary
