## The Documentation Window

- Primary documentation
	- References documentation for Cocoa classes and other symbols
- Secondary documentation
	- Consists of older guides, sample code, and technical notes

## Class Documentation Pages

- Function declaration with a comment

```swift
/**
Many people would like to dog their cats. So it is *perfectly*
reasonable to supply a convenience method to do so:

* Because it's cool.
* Because it's there.

* Parameter cats: A string containing cats

* Returns: A string containing dogs
*/

func dogMyCats(_ cats:String) -> String {
    return "Dogs"
}
```
![[Pasted image 20260526144105.png]]
## Quick Help

- DocC
	- Comment based documentation
- Build documentation

```swift
/// Degree of coolness.
public enum Temp {
    /// So cool you could plotz.
    case frigid
    /// Kind of cool.
    case lukewarm
    /// Not particularly cool.
    case boiling
}

/// Expression of coolness.
public struct Cool {
    var temp : Temp
    /// Changes our coolness.
    /// - Parameter to: The Temp you'd like it to be.
    public mutating func changeTemp(to newValue: Temp) {
        self.temp = newValue
    }
}
```
## Documenting Frameworks and Packages

- Customize a landing page with markdown

```siwft
# ``Coolness``

This framework is extremely cool.

## Overview

How cool can you get?

Very cool!

## Topics

### Basic

These are basic.

- ``Cool``

### Advanced

These are advanced.

- ``Temp``
```

- A explanatory article files
- Resources folder can hold images to include in documentation
- Construct interactive tutorials

## Symbol Declarations

- A symbol is a declared term
	- Function, variable, or object type
## Header Files

- A header file can be a useful form of documentation
## Sample Code
## Internet Resources