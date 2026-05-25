
- Xcode project
	- Source for an app
	- Entire collection of files and settings used to construct the app

## New Project

## The Project Window

- Xcode project
	- Source files that are to be compiled
	- `.storyboard` or `.xib` files
		- Graphically expressing interface objects to be instantiated
	- Any resources
	- All settings
	- Any frameworks that the code will need when it runs

GUI
- Navigator pane
- Editor pane
- Inspector pane
- Debug pane

### The Navigator Pane

- Column of information
### The Inspector Pane

- Column at the right of the project window
- Contains inspectors that provide information about the current selection or its settings
### The Editor

- Main window for writing code

## Project File and Dependents

- First item in the Project navigator is the project itself (Empty Window)

### Contents of the Project Folder

- `Window.xcodeproj`
	- Project file
	- How to build the project
- `LaunchScreen.storyboard`
	- Localization
- `Assets.xcassets`
	- Asset catalog
	- Added resources
### Groups

- A group might or might not correspond to a folder on disk in the project folder
- A group that corresponds to a folder on disk is a folder-linked group
	- Solid folder icon
	- Empty Window Group
- A group exists purely within the Project navigator
	- Icon marked with a little triangle in its lower left corner

## The Targets

- A target is a collection of parts along with rules and settings for how to build a project from those parts
- App target
	- Target that you use to build and run the app
- Add further targets to a project
	- Add units tests
	- Write an application extension
	- Write a library, custom framework

### Build Phases

- Comple sources
- Copy bundle resources
### Build Settings

- Build phases are one aspect of how a target known how to build the app
- The other is build settings
### Configurations

- Default
	- Used through development process
	- Write and run app
- Release
	- Late stage testing
	- Check performance
	- Archiving the app to be submitted to the App store
### Schemes and Destinations

- A scheme unites a target with a build configuration
- Destination is the machine that the app can run on

## From Project to Built App

- App bundle
	- Empty window
		- Binary app
		- Executable
	- `Main.storyboardc`
		- App interface
	- `LaunchScreen.storyboardc`
		- Contains interface that is displayed on launch
	- `Assets.car`
		- Asset catalog
	- Info.plist
		- Property list file

### Build Settings

### Property List Settings

### Nib Files

- A nib file is a file containing a description of a piece of user interface in a compiled format
### Resources

- Resources are ancillary files embedded in the app bundle
	- Add resources
		- Project navigator
		- Asset catalog

#### Resources in the Project Navigator

- 
#### Resources in the Asset Catalog
### Code Files
### Framework and SDKs
### Swift Packages

#### Adding a Package
#### Creating a Package
#### Customizing a Package



## App Launch Process

### The Entry Point
### How an App Gets Going
### App Without A Storyboard


## Renaming Parts of a Project


