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

```swift
UIView.animate(withDuration:0.4,
	animations: {
		() -> () in
		self.myButton.frame.origin.y += 20
	},
	completion: {
		(finished:Bool) -> () in
		print("finished: \(finished)")
	}
)
```

### Anonymous Function Abbreviated Syntax

- Omission of the return type
	- Omit the arrow and the specification of the return type

```swift
UIView.animate(withDuration:0.4,
    animations: {
        () in
        self.myButton.frame.origin.y += 20
    }, completion: {
        (finished:Bool) in
        print("finished: \(finished)")
})
```

- Omit the `in` if there are no parameters

```swift
UIView.animate(withDuration:0.4,
    animations: {
        self.myButton.frame.origin.y += 20
    }, completion: {
        (finished:Bool) in
        print("finished: \(finished)")
})
```

- Omit the parameter types

```swift
UIView.animate(withDuration:0.4,
    animations: {
        self.myButton.frame.origin.y += 20
    }, completion: {
        (finished) in 
        print("finished: \(finished)")
})
```

- Omit the parentheses

```swift
UIView.animate(withDuration:0.4,
    animations: {
        self.myButton.frame.origin.y += 20
    }, completion: {
        finished in // *
        print("finished: \(finished)")
})
```

- Omit `in` expression where there are parameters
	- Use magic names, `$0`

```swift
UIView.animate(withDuration:0.4,
    animations: {
        self.myButton.frame.origin.y += 20
    }, completion: {
        print("finished: \($0)") 
})
```

- Omit parameter names

```swift
UIView.animate(withDuration:0.4,
    animations: {
        self.myButton.frame.origin.y += 20
    }, completion: {
        _ in 
        print("finished!")
})
```

- Omit function argument labels
	- Trailing closure syntax
	- Pass the anonymous function argument outside the call's parentheses with no label

```swift
UIView.animate(withDuration:0.4,
    animations: {
        self.myButton.frame.origin.y += 20
    }) { 
        _ in
        print("finished!")
}
```

- Pass multiple anonymous function arguments
	- First anonymous function takes no label
	- Remaining function do have labels, with no comma

```swift
UIView.animate(withDuration:0.4) { 
    self.myButton3.frame.origin.y += 20
} completion: { // *
    _ in
    print("finished")
}
```

- Omit calling function parentheses
	- If there is a trailing closure and no parameters

```swift
func doThis(_ f:()) {
	f()
}

doThis {
	print("Hello")
}
```

- Omit `return`

```swift
func greeting() -> String {
    return "Howdy"
}
func performAndPrint(_ f:()->String) {
    let s = f()
    print(s)
}
performAndPrint {
    greeting() // meaning: return greeting()
}
```

- `map(_:)`
	- Takes an array as a function

```swift
let arr = [2, 4, 6, 8]

func doubleMe(i:Int) -> Int {
	return i*2
}

let arr2 = arr.map(doubleMe)

// Anonymous function

let arr2 = arr.map ({
	(i:Int) -> Int in
	return i*2
})

// Omit parentheses

let arr2 = arr.map {$0*2}
```

## Define and Call

- Define an anonymous function and call it

```swift
{

}()
```

- The curly braces define an anonymous function body
- Parentheses call that anonymous function
- Action can be taken at the point where is is needed

```swift
content.addAttribute(
    .paragraphStyle,
    value: {
        let para = NSMutableParagraphStyle()
        para.headIndent = 10
        para.firstLineHeadIndent = 10
        // ... more configuration of para ...
        return para
    }(),
    range:NSRange(location:0, length:1))
```

## Closures

- Swift functions are closures
- Capture reference to external variables in scope within the body of the function

```swift
class Dog {
    var whatThisDogSays = "woof" 
    func bark() {
        print(self.whatThisDogSays) 
    }
}
```

- `whatThisDogSays` is external to the function
	- Declared outside the body of the function
- `bark()`
	- Code inside refers to the external variable
- A function is a closure and it captures external variables referred to in its body

### How Closures Improve Code

- Functions are closures to make code more general

```swift
let sz = CGSize(width:45, height:20)
let image = imageOfSize(sz) {
    let p = UIBezierPath(
        roundedRect: CGRect(origin:CGPoint.zero, size:sz),
        cornerRadius: 8)
    p.stroke()
}

func makeRoundedRectangle(_ sz:CGSize) -> UIImage {
    let image = imageOfSize(sz) {
        let p = UIBezierPath(
            roundedRect: CGRect(origin:CGPoint.zero, size:sz),
            cornerRadius: 8)
        p.stroke()
    }
    return image
}
```

### Function Returning Function

- Takes a CGSize parameter and returns a `() -> UIImage`
	- Returns a function with no parameters
- Return a function

```swift
func makeRoundedRectangleMaker(_ sz:CGSize) -> () -> UIImage { 
    func f () -> UIImage { 
        let im = imageOfSize(sz) {
            let p = UIBezierPath(
                roundedRect: CGRect(origin:CGPoint.zero, size:sz),
                cornerRadius: )
            p.stroke()
        }
        return im
    }
    return f 
}

let maker = makeRoundedRectangleMaker(CGSize(width:45, height:20))
self.iv.image = maker()
```

### Closure Setting a Captured Variable

- If the closure captures a reference to a variable outside itself, and it is seeable, then the closure can set the variable

```swift
func pass100 (_ f:(Int) -> ()) {
	f(100)
}

var x = 0
print(x)

func setX(newX:Int) {
	x = newX
}

pass100(setX)
print(x)
```

- The `pass100` function has reached inside the function to change the value of `x`

### Closure Preserving Captured Environment

- When a closure captures its environment, it preserves that environment even if nothing else does

```swift

func countAdder(_ f: @escaping () -> ()) -> () -> () {
	var ct = 0
	return {
		ct = ct + 1
		print("count is \(ct)")
		f()
	}
}
```

- The function accepts a function as its parameter and returns a function
- The function that is returns calls the function that is accepts

```swift
func greet() {
	print("Hello")
}

let countedGreet = countAdder(greet)
countedGreet() // ?
countedGreet() // ?
countedGreet() // ?

// output
count is 1
howdy
count is 2
howdy
count is 3
howdy
```

- `ct` variable must be declared outside the anonymous function

### Escaping Closures

- Escaping closure
	- A function passed around as a value
	- Preserved for later execution
	- `@escaping`

```swift
func funcCaller(f:() -> ()) {
	f()
}
```

- Create the function internally
- The returning function is an escaping closures

```swift
func funcMaker() -> () -> () {
	return { print("hello") }
}
```

- Make the type of the incoming parameter as escaping, and the compiler will be prompted to execute it later

```swift
func funcPasser(f:@escaping () -> ()) -> () -> () {
	return f
}
```

- `self` is used as a reference capture

```swift
let f1 = funcPasser {
    print(view.bounds) // compile error, because self.view is implied
}
let f2 = funcPasser {
    print(self.view.bounds) // ok
}
```

### Capture Lists

- Use square brackets outside an anonymous function to refer to a variable without capture
- Capture list
- Use `in` expression to capture list

```swift
var x = 0
let f : () -> () = {
	print(x)
}
f()
x = 1
f()

// capture list
let f : () -> () = { [x] in
	print(x)
}
f()
x = 1
f()
```

- The capture lists prints `0` both times
- Sets `x` as a constant
- Capture list expression to another name

```swift
self.undoer.registerUndo(withTarget: self) {
    [oldCenter = self.center] myself in
    myself.setCenterUndoably(oldCenter)
}
```

## Curried Functions

```swift
func makeRoundedRectangleMaker(_ sz:CGSize, _ r:CGFloat) -> () -> UIImage {
    return {
        imageOfSize(sz) {
            let p = UIBezierPath(
                roundedRect: CGRect(origin:CGPoint.zero, size:sz),
                cornerRadius: r)
            p.stroke()
        }
    }
}

let maker = makeRoundedRectangleMaker(CGSize(width:45, height:20), 8)


// no parameters
func makeRoundedRectangleMaker(_ sz:CGSize) -> (CGFloat) -> UIImage {
    return { r in
        imageOfSize(sz) {
            let p = UIBezierPath(
                roundedRect: CGRect(origin:CGPoint.zero, size:sz),
                cornerRadius: r)
            p.stroke()
        }
    }
}

let maker = makeRoundedRectangleMaker(CGSize(width:45, height:20))
self.iv.image = maker(8)

// or
self.iv.image = makeRoundedRectangleMaker(CGSize(width:45, height:20))(8)

```

- When a function returns a function that takes a parameter in this way is is called a curried function


## Function References and Selectors

- A bare name is a function reference
- The lack of parentheses make is clear that this is a reference, and not a call