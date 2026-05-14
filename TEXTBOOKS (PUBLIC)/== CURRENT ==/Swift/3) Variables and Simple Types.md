
- Each variable is explicitly and formally declared
- A variable name refers to a value
## Variable Scope and Lifetime

- Global variables
	- A variable declared at the top level of a Swift file
	- Visible to other files in the same module
- Properties
	- A property is a variable declared at the top level of an object type declaration
		- Instance properties
		- Static/class properties
			- `static`
			- `class`
	- Visible only by way of the object
- Local variables
	- Declared inside a function body

## Variable Declaration

- `let`
	- Constant values
- `var`
	- Can change values

- Variable declaration is followed by initialization
	- Not a requirement
	- Must have a type
	- Type cannot be changed, only reassigned
- Explicit
	- `var x : Int`
- Implicit
	- `var x = 1`

- A local variable should be initialized when declared
- Conditional initialization
- Declare variable with placeholder values

```swift
var bti : UIBackgroundTaskIdentifier = .invalid
bti = UIApplication.shared.beginBackgroundTask {
    UIApplication.shared.endBackgroundTask(bti)
}
```

## Computer Variable Initialization

- Define and call anonymous function

```swift
let timed : Bool = {
	if val == 1 {
		return true
	} else {
		return false
	}
} ()
```

- At the time of initializing the instance property, there is no instance
- A define and call anonymous function cal refer to `self`

## Computed Variables

- The previous variables were stored
- Other variables can be computer
	- Setter
	- Getter

```swift
var now : String {
	get {
		return Date().description
	}
	set {
		print(newValue)
	}
}

now = "Hello"
print(now)
```

### Computer Properties

- Facade for a longer expression

```swift
var mp : MPMusicPlayerController {
    MPMusicPlayerController.systemMusicPlayer
}
var nowPlayingItem : MPMediaItem? {
    self.mp.nowPlayingItem
}
```

- Facade for an elaborate calculation
	- Method expresses a process

```swift
var authorOfItem : String? {
    guard let authorNodes =
        self.extensionElements(
            withXMLNamespace: "http://www.tidbits.com/dummy",
            elementName: "app_author_name")
        else {return nil}
    guard let authorNode = authorNodes.last as? FPExtensionNode
        else {return nil}
    return authorNode.stringValue
}
```

- Facade for storage
	- A public computer variable is backed by a private stored variable

```swift
private var _p : String = ""
var p : String {
    get {
        self._p
    }
    set {
        self._p = newValue
    }
}
```

### Property Wrappers

- Property wrapper is declared as a type marked with the `@propertyWrapper` attribute
	- Must have a `WrappedValue` computed property

```swift
@propertyWrapper struct Clamped {
    private var _i : Int = 0
    var wrappedValue : Int {
        get {
            self._i
        }
        set {
            self._i = Swift.max(Swift.min(newValue,5),0)
        }
    }
}

@Clamped var p
```

- Declare a computer property marked with a custom attribute
	- Does not need to be initialized, have type declaration, or a getter/setter

## Setter Observers

- Setter observers
	- Functions that are called before and after other code sets a stored variable
	- Not called when the stored variable is initialized
	- `willSet
	- `didSet`

```swift
var s = "whatever" { 1
    willSet { 2
        print(newValue) 3
    }
    didSet { 4
        print(oldValue) 5
        // self.s = "something else"
    }
}
```
- Visible interface to reflect the state of objects

```swift
var angle : CGFloat = 0 {
	didSet {
		self.tranform = CGAffineTransform(rotationAngle: self.angle)
	}
}
```

## Lazy Initialization

- Initial value is not evaluated and assigned until running code accesses the variable's value

- Global variables
	- Automatically lazy
- Static properties
	- Automatically lazy
- Instance properties
	- Not lazy by default
	- `lazy var`
- Local variables
	- Can be declared with `lazy var`


### Singleton

- Lazy initializtion is often used to implement singleton
	- A pattern where all code is able to get access to a single shared instance of a certain class

```swift
class MyClass {
	static let shared = MyClass()
}

MyClass.shared
```

- Singleton instance is not created until the first time other code called `shared`

### Lazy Initialization of Instance Properties

- A lazy initializer can refer to the instance
- Is not guaranteed  to run until after the instance fully exists
- Initialize a lazy instance property with a define-and-call anonymous function with `self`

```swift
lazy var prog : UIProgressView = {
	let p = UIProgressView(progressViewStyle: .default)
	p.alpha = 0.7
    p.trackTintColor = UIColor.clear
    p.progressTintColor = UIColor.black
    p.frame =
        CGRect(x:0, y:0, width:self.view.bounds.size.width, height:20) // legal
    p.progress = 1.0
    return p
}()
}
```

## Build-In Simple Types

### Bool

- Bool object type
	- A struct
	- Two values
		- `true` and `false`
	- Nothing else in Swift is implicitly coerced to or treated as a Bool
- `!, &&, ||`
- Use `toggle()` as a bool negation

### Numbers

#### Int

- Int object type
	- Struct
	- Integer between `Int.min` and `Int.max` inclusive
	- $-2^{63}$ and $2^{63} - 1$
	- 64 bit words
	- A numeric literal
	- Internal underscores are legal
- Int literal types
	- `0b, 0o, 0x`
	- Binary, octal, or hexadecimal digits
- Negative numbers are stored in the two's complement format

#### Double

- Double object type
	- Struct
	- Floating point number
	- Precision of about 15 decimal places
- Exponents
- Static properties
	- `Double.infinity`
	- `Double.pi`
	- `isZero`

#### Number Coercion

- Coercion is the conversion of a value from one type to another
- Instantiation
	- Instantiate a Double with the Int in the parentheses

```swift
let i = 10
let x = Double(i)
print(x) // 10.0, a Double
let y = 3.8
let j = Int(y)
print(j) // 3, an Int
```

- Implicit coercion of literals when assigns to variables or passed as arguments

#### Other numeric types

- Integer
	- Int8
	- Int16
	- Int32
	- Int64
	- UInt8
	- IUnt16
	- UInt32
	- UInt64
- Float
	- Float16
	- Float80
	- CGFloat (Core Graphics framework)
- Type aliases
	- CDouble
	- Long
	- TimeInterval

- CGFloat
	- Resolves to the size of Float or Double
		- Depending on architecture
- Failable initializer
	- Code is legal, but will crash at runtime
- `Clamping`
	- Always succeeds, because an out of range value is forced to fall within range

#### Arithmetic Operations

- Logical operators
	- `_, -, *, /, %`
- Bitwise operators
- `&, |, ^, ~, <<, >>`

- Compound assignment
	- `+=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=`

- Math methods
	- `import Numerics`
	- `abs, min, max`
	- `squareRoot()`
	- `rounded()`
	- `random(in:)`

#### Comparison

- `==, !=, <, <=, >, >=`

### String

- String object type
- A literal delimited by double quotes
- Unicode
	- `\u{}`


- Escape characters
	- `\n, \t, \", \\`
- Regular expressions

- String interpolrat

### Character and String Index
### Range
### Tuple
### Optional
#### Unwrapping an Option
#### Implicitly Unwrapped Optional
### Keyword `nil`
### Optional Chains
#### Optional map and flatMap

#### Comparison with Optional
