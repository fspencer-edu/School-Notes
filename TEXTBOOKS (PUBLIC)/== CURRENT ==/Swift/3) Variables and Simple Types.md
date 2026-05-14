
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

## Setter Observers
## Lazy Initialization
## Build-In Simple Types
