## Environmental Dependencies

- Build-time dependencies
	- Version
	- Type of destination
	- Custom compilation
	- Action that causes the build
- Run time dependencies
	- App environment
	- Resources

### Conditional Compilation

```swift
#if condition
    statements
#elseif condition
    statements
#else
    statements
#endif

swift(>=5.5)
compiler(>=5.5)
targetEnvironment(simulator)
canImport(UIKit)
```
- Conditions are treated as Bools
- Combined with Boolean logic operators
- Define an Active Compilation Conditions settings for that configuration

### Build Action

- Development Assets build settings lets you specify one or more paths for resources that won't be included in an archive build

### Permissible Runtime Environment

- Build settings
	- Device type
		- iPhone
		- iPad
		- Universal app
	- iOS Deployment Target

### Backward Compatibility

- Availability check

```swift
if #available(iOS 10.0, *) {
    let r = UIGraphicsImageRenderer(size:CGSize(width:10,height:10))
} else {
    // Fallback on earlier versions
}
```

### Device Type

- `Info.plist`

### Arguments and Environment Variables

- Edit the scheme and go to the arguments tab
- Arguments and environmental variables are configured for the run action are present when you build and run, not while testing

## Version Control

- Security
- Publication
- Collaboration
- Confidence

- GitHub
	- Create a remote repository
	- Clone a remote repository
	- Manage pull requests

## Editing and Navigating Your Code

### Text Editing Preferences

#### Display
#### Editing
#### Indentation
### Multiple Selection

### Code Completion and Placeholders
### Snippets
### Refactoring and Code Actions

- Rename
- Extract to method
- Extract to variable
- Add missing protocol requirements
- Generate memberwise initializer
- Add equatable conformance
- Convert function to async
	- Add async alternative
	- Add async wrapper
- Convert to switch statement
- Expand switch cases
### Fix-it and Live Syntax Checking

### Navigation

- Project navigator
- Symbol navigator
- Jump bar
- Editor panes
- Assistant panes
- Document tabs
- Jump to definition
- Open quickly
- Breakpoints
- Minimap
### Finding

- Editor level find
- Global find

## Running in the Simulator

## Debugging

### Caveman Debugging

- Consists of altering code, temporarily
- Add code to log informative messages
	- `#file`
	- `#line`
	- `#column`
	- `#function`

#### Print

```swift
print("view did load")
print(self.view)
```
#### Dump

```swift
dump(self)
```
#### Logger

- `print` and `dump` produce no output when the app is launched independently of Xcode
- `NSlog` C function
	- A format string is a string containing symbols called format specifiers

```swift
NSLog("the view: %@", self.view)
```

- OSLog
	- A Swift native type, acts as a comfortable facade for OSLOG

```swift
import os
let mylog = Logger(subsystem: "com.neuburg.matt", category: "testing")

mylog.log("this is a test")
```

#### Aborting
### The Xcode Debugger

#### Breakpoints
#### Paused at a breakpoint

## Testing

- A test is code to exercise your app and make sure that is works as expected
	- Unit test
	- Interface tests

- Tests are bundled in a separate target
- A test class is a subclass of XCTestCase
- A test method is an instance method of a test class
- Test methods
	- Assertions
	- Throws

- Each test method runs in its own separate test class instance

### Unit Tests

- Unit test need to see into the target to be tested

```swift
func dogMyCats(_ s:String) -> String {
    return ""
}

func testDogMyCats() {
    let input = "cats"
    let output = "dogs"
    XCTAssertEqual(output,
        self.viewController.dogMyCats(input),
        "Failed to produce \(output) from \(input)")
}
```
### Interface Tests

```swift
let app = XCUIApplication()
app.staticTexts["Hello"].tap()
app.alerts["Howdy!"].scrollViews.otherElements.buttons["OK"].tap()
```

#### Persisting Screenshots

- Screenshots taken automatically during the UI test can be used for other purposes
	- Marketing material
	- Submission to the App store
	- Localization

```swift
let screenshot = XCUIApplication().screenshot()
let attachment = XCTAttachment(screenshot: screenshot)
attachment.lifetime = .keepAlways
attachment.name = "OpeningScreen"
self.add(attachment)

```
#### Interface testing and accessibility

### Test Plans

- Scheme's Test action
	- Determine the complete set of tests to run
- Test plans are written in a text file (JSON)
- 
### Massaging the Reports

## Clean

- During repeated testing and debugging, clean target
- Existing builds will be removed and caches will be cleared

## Running on a Device

### Obtaining a Developer Program Membership
### Signing an App

### Automatic Signing
### Manual Signing
### Running the App
### Managing Development Certificates and Devices

## Profiling

### Gauges

- The gauges in the Debug navigator are operating whenever you build and run your app
	- GPU
	- Memory
	- Disk
	- Network
### Memory Debugging

- Memory debugging lets you pause your app and view a graphical display of object hierarchy at that moment
### Instruments

## Localization

### Creating Localized Content
#### Exporting

#### Editing

#### Importing
### Testing Localization
## Distribution

- Distribution means sharing your built app with users for running on their devices
	- Ad Hoc distribution
	- App Store Connect distribution
		- TestFlight testing
		- App Store sale

### Making an Archive

- Build an archive of your app
	- Distribution
	- Reproduction
	- Symbolication
### The Distribution Certificate

- Distribution certificate is required for distributing your app to other users
### The Distribution Profile
### Distribution for Testing
#### Ad Hoc Distribution
### Final App Preparations

#### Icons in the app
#### Marketing icon
#### Launch Image
### Screenshots and Video Previews

### Property List Settings
### Submission to the App Store
### 
### 

Tofu & Veggies pad tai
Pad Cashew Nut
Shrimp Cold Rolls or Pineapple Fried Rice