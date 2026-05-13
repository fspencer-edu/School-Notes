## Function Parameters and Return Value

- A function expects parameters, and produces a result

```swift
func sum (_ x:Int, _ y:Int) -> Int {
	let result = x + y
	return result
}

let z = sum(4, 5)
```

- Parameter variables are internal to the function
- A function call uses the name of the function and passes the parameters in parentheses
- Values are called arguments

## Void Return Type and Parameters


- A function without a return type
	- Return a void, empty, or no arrow operator
	- A function with no returns is purely side effects
```swift
func say1(_ s:String) -> Void { print(s) }
func say2(_ s:String) -> ()) { print(s) }
func say3(_ s:String) { print(s) }
```
- Function without parameters

```swift
func greet() -> String { return "hello" }
```

## Function Signature

- Signature of a function
	- Characterizes all functions that have this number of parameters
	- Includes parameter list and return type

```swift
(Int, Int) -> Int
```

## External Parameter Names

- A function can externalize the names of its parameters
- Must appear in a call to the function as labels to the arguments
	- Clarifies the purpose of each argument
- By default, all parameter names are externalized automatically
	- Internal names are used as the external names
- Change the name of an external parameter
	- Precede the internal name with external name and a space
	- Suppress the externalization of a parameter
		- Underscore and space

```swift
func echoStrring(_ s:String, times:Int) -> String {
	var result = ""
	for _ in 1...times { result += s }
	return result
}

let s = echoString("hi", times:3)
```

## Overloading

- Overloading is legal
- Two function with the same name, including external parameter names, can coexist with different signatures

```swift
func say (_ what:String) {
}
func say (_ what:Int) {
}

func say() -> String {
    return "one"
}
func say() -> Int {
    return 1
}
```

- Swift has strict typing
- Calling the function must expect a specific return type

```swift
let result: String = say()
```

- Disambiguate  between overloads in a method call using the name of the method, the keyword `as`, and the signature

```swift
let result = (say as () -> String)()
```

## Default Parameter Values

- A parameter can have a default value
- Called can omit function arguments
- Append `=` to a default value after the parameter type

```swift
class Dog {
	func say(_ s:String, times:Int = 1) {
		for _ in 1...times {
			print(s)
		}
	}
}
```

## Variadic Parameters

- A parameter can be variadic
- Called can supply as many arguments values of this parameter's type

```swift
func sayStrings(_ array:String ...) {
	for s in array { print(s) }
}

sayStrings("hello", "there")

print("Manny", 3, true) // Manny 3 true
```

- Remaining parameters have default values types and can be omitted
- Add a `seperator:` and `terminator:` to dictate output details

```swift
print("Manny", "Moe", separator:", ", terminator:", ")
print("Jack")
```

- A function can declare a max of one variadic parameter
- New versions can use more than one
- There is no way to convert an array into a comma separated list of arguments
	- Splatting in Ruby
- An initializer is a function, that can omit or include parameters
	- Overload default initializer parameter values

```swift
let s1 = String(42)
et s2 = String(repeating: "hello", count: 2)
```

## Ignored Parameters

- A local name with an underscore is ignored

```swift
func say(_ s:String, times:Int, loudly _:Bool) {

say("hi", times:3, loudly:true)
```

## Modifiable Parameters

- A parameter is a local variable

```swift
func removeChar(_ c:Character, from s:String) -> Int {
	var s = s
	var howMany = 0
	while let ix = s.firstIndex(of:c) {
		s.remove(at:ix)
		howMany += 1
	}
	return howMany
}

let s = "hello"
let result = removeChar("l", from:s)
```

- To alter the original value of an argument passed
	- Type of the parameter must be declared `inout`
	- The variable holding the value is modified with a `var`
	- Instead of passing the variable as an argument, pass its address with `&`

```swift
func removeChar(_ c:Character, from s: inout String) -> Int {
	var howMany = 0
	while let ix = s.firstIndex(of:c) {
		s.remove(at:ix)
		howMany += 1
	}
	return howMany
}

var s = "hello"
let result = removeChar("l", from:&s)
```

- When a function with an `inout` parameter is called, the variable whose address was passed as argument to the parameter is always set

## Calling Objective-C with Modifiable Parameters

- Cocoa APIs is written in C and Objective-C
	- `UnsafeMutablePointer`

```swift
func getRed(_ red: UnsafeMutablePointer<CGFloat>,
    green: UnsafeMutablePointer<CGFloat>,
    blue: UnsafeMutablePointer<CGFloat>,
    alpha: UnsafeMutablePointer<CGFloat>) -> Bool
    
let c = UIColor.purple
var r : CGFloat = 0
var g : CGFloat = 0
var b : CGFloat = 0
var a : CGFloat = 0
c.getRed(&r, green: &g, blue: &b, alpha: &a)
```

- Sometime Cocoa will call a function
	- Cannot assign directly to it
	- Assign to the `pointee` property

```swift
func popoverPresentationController(
    _ popoverPresentationController: UIPopoverPresentationController,
    willRepositionPopoverTo rect: UnsafeMutablePointer<CGRect>,
    in view: AutoreleasingUnsafeMutablePointer<UIView>) {
        view.pointee = self.button2
        rect.pointee = self.button2.bounds
}
```

## Reference Type Modifiable Parameters

- Function can modify a parameter without declaring it as `inout`
- Parameter is an instance of a class

```swift
class Dog {
 var name = ""
}

func changeName(of d:Dog, to newName:String) {
	d.name = newName
}

let d = Dog()
d.name = "Fido"
print(d.name)
changeName:of:d, to:"Rover")
print(d.name)
```

- Class instances are mutable
- Classes are reference types
- Object type flavours are value types
- When an instance of a struct is passed as an argument, is produces a separate copy of the struct isntance
- When a instance of a class is passed, the reference to the class instance is passed

## Function in Function

- A function declared in the body of a function
	- Local function

## Recursion

- A function that call call itself

```swift
func countDownFrom(_ ix:Int) {
	print(ix)
	if ix > 0 {
		countDownFrom(ix - 1)
	}
}

countDownFrom(5)
```

## Function as Value

- Functions are first-class citizen
	- Function can be used wherever a value is used
	- A function can be assigned to a variable
	- Passed as an argument
	- Returns as the result of a function
- Assign a value to a variable or pass a value into or out of a function only if it is the right type of value

```swift
func doThis(_ f:() -> ()) {
	f()
}

func whatToDo() {
	print("Hello")
}

doThis(whatToDo)
```

- Takes one function as a parameter, and has no return type

```swift
let size = CGSize(width:45, height:20)
UIGraphicsBeginImageContextWithOptions(size, false, 0) 
let p = UIBezierPath(
    roundedRect: CGRect(x:0, y:0, width:45, height:20), cornerRadius: 8)
p.stroke() 
let result = UIGraphicsGetImageFromCurrentImageContext()! 
UIGraphicsEndImageContext() 
```

- Generates an image of a rounded rectangle

```swift
func imageOfSize(_ size:CGSize, _ whatToDraw:() -> ()) -> UIImage {
    UIGraphicsBeginImageContextWithOptions(size, false, 0)
    whatToDraw()
    let result = UIGraphicsGetImageFromCurrentImageContext()!
    UIGraphicsEndImageContext()
    return result
}

func drawing() {
    let p = UIBezierPath(
        roundedRect: CGRect(x:0, y:0, width:45, height:20),
        cornerRadius: 8)
    p.stroke()
}
let image = imageOfSize(CGSize(width:45, height:20), drawing)

func whatToAnimate() { // self.myButton is a button in the interface
    self.myButton.frame.origin.y += 20
}
func whatToDoLater(finished:Bool) {
    print("finished: \(finished)")
}
UIView.animate(withDuration:0.4,
    animations: whatToAnimate, completion: whatToDoLater)
```

- `\` is used for string interpolation
- A function to be passed is called a handler or a block
- Use type aliases to give a function type a name

```swift
typealias VoidVoidFunction = () -> ()

func dothis(_ f:VoidVoidFunction) {
	f()
}
```

## Anonymous Functions

- Use anonymous functions
	- Create the function body, with no function declaration
	- Express the function's parameter list and return type as the first thing inside the curly braces, followed by `in`

```swift
func whatToAnimate() {
	self.myButton.frame.origin.y += 20
}

{
	() -> () in
	self.myButton.frame.origin.y += 20
}

func whatToDoLatter(finished:Bool) {
	print("finished: \(finished)")
}

{
	(finished:Bool) -> () in
	print("finished: \(finished)")
}
```

### Anonymous Functions Inline

