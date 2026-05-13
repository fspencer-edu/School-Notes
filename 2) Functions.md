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

```swift
func say1(_ s:String) -> Void { print(s) }
func say2(_ s:String) -> ()) { print(s) }
func say3(_ s:String) { print(s) }
```

- A function without a return type
	- Return a void, empty, or no arrow operator
- A function with no returns is purely side effects
- Function without