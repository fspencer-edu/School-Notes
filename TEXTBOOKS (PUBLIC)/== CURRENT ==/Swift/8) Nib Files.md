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
	- 
### Canvas
### Inspectors

## Loading a Nib

### Loading a View Controller Nib
### Loading a Main View Nib
### Loading a View Nib Manually

## Connections

### Outlets
### The Nib Owner
### Automatically Configured Nibs
### Misconfigured Outlets

#### Outlet - Property name mismatch
#### No outlet in the nib
#### No view outlet
### Deleting an Outlet
### More Ways to Create Outlets
### Outlet Collections
### Action Connections
### More Ways to Create Actions
### Misconfigured Actions
### Connections Between Nibs

## Additional Configuration of Nib-Based Instances
