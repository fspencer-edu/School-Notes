- App's visible interface consists entirely of views
- A view is an interface object
	- Contain subviews
	- All views are placed by a view controller
		- Main view

```swift
let v = UIView()
```
- Nib
	- A file, in a special format, consisting of instructions for creating and configuring instances, primarily UIView instances
- Xcode includes a graphical design environment
	- Nib editor

## The Nib Editor Interface

- Bulk of the editor is the canvas
- Document outline

### Document Outline

- Shows the hierarchical relationship between the objects in the nib
- Storyboard file
	- Consists of scenes
	- A single view controller with ancillary material
- View controller
	- Manages an interface object
	- A view that serves as its main view
- Nib objects
	- View controller, main view, and subviews are turned into instances
- Proxy objects
	- Objects that already exist
	- First Responder
	- Exit tokens

- `.xib` file
	- No scenes

### Canvas

- Provides a graphical representation of a view and its subviews
- `rootViewController`

### Inspectors

- Identity
	- Object's class
- Attributes
	- Properties and methods that are used to configure the object in code
	- Sections of object's class inheritance
- Size
	- `frame` property
- Connections
	- Connections are outlets and actions

## Loading a Nib

- A nib file is a collection of potential instances
- Become actual instances, when the app is running, and nib is loaded
- Nib transformed into instances
- Nib objects are turned into instances, and those instances are handed over to the running app
- Nib file can be loaded multiple times

### Loading a View Controller Nib

- A nib containing a view controller will come from a storyboard
- A storyboard is a collection of scenes
- Loaded automatically or manually
	- Automatic
		- Launch time
		- Segue is performed
	- Manual
		- `instantiateInitialViewController`
		- `instantiateViewController(withIdentifier:)
`
### Loading a Main View Nib

- Every view controller has a main view
- A view controller, when it is instantiated, lacks its main view
- View controller loads its view lazily
	- View controller in a storyboard
	- View controller instantiated in code
### Loading a View Nib Manually

## Connections

- A connection is a directional linkage in the nib editor running from one object to another
- Sources and destination
- Outlet connections
- Action connections
### Outlets

- An outlet is a connection that has a name
- Source object and the destination object are no longer potential objects in a nib
- Runtime looks in the outlet's source object for an instance property with the same name as the outlet, and assigns the destination object to that property

![[Pasted image 20260526141922.png]]

### The Nib Owner

- Nib editor permits an outlet to be created, using the nib owner object
- Storyboard scene
	- Nib owner is the view controller
- `.xib`
	- Nib owner is a proxy object

### Automatically Configured Nibs

- When a view controller gets its main view from a nib automatically, everything works
	- Instance property
	- Nib owner class
	- Outlet

### Misconfigured Outlets

#### Outlet - Property name mismatch

#### No outlet in the nib
#### No view outlet
### Deleting an Outlet
### More Ways to Create Outlets
### Outlet Collections
### Action Connections

- An action connection is a way of given one object in a nib a reference to another
- It's not a property reference
- Message sending reference
- An action is a message emitted anatomically by a Cocoa UIControl interface object, sent to another object when the user does something to it
- Control objects
	- Control event
	- Action
	- Target

```swift
@IBAction func buttonPressed(_ sender: Any) {
    let alert = UIAlertController(
        title: "Howdy!", message: "You tapped me!", preferredStyle: .alert)
    alert.addAction(
        UIAlertAction(title: "OK", style: .cancel))
    self.present(alert, animated: true)
}
```

### More Ways to Create Actions
### Misconfigured Actions
### Connections Between Nibs

## Additional Configuration of Nib-Based Instances
