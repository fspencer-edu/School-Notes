# Types of Tests

<img src="/images/Pasted image 20260428103803.png" alt="image" width="500">

- Technology facing
	- Aid developers
	- Performance tests
	- Small scoped unit tests
	- Automated
- Business facing
	- Large scoped
	- End-to-end tests
	- UAT

# Test Scope

- Test Pyramid
	- Unit
	- Service
	- UI

<img src="/images/Pasted image 20260428103942.png" alt="image" width="500">

## Unit Tests

- Test a single function or method call
- Generated as a side effect of test-driven design (TDD)
- Technology facing
- Catch most bugs
- Fast feedback
- Refactoring of code, or restricting

<img src="/images/Pasted image 20260428104139.png" alt="image" width="500">

## Service Tests

- Bypass the user interface and tests services directly
- Testing a collection of classes the provide a service to the UI
- Cover more scope than simple unit test
- Stub database


<img src="/images/Pasted image 20260428104245.png" alt="image" width="500">

## End-to-End Tests/UI

- Run tests against the entire system
- Driving a GUI through a browser
- Uploading a file
- More complex to implement

<img src="/images/Pasted image 20260428104431.png" alt="image" width="500">

## Trade-Offs

- Test scope increases the higher the pyramid
	- Confidence in functionality being tested increases
	- Feedback cycle time increases

## How Many?

- Order of magnitude more tests as you descend the pyramid
- Different type of automated tests
- Test snow cone (inverted pyramid)
	- Little to no small-scoped tests
	- All the coverage in large-scoped tests
	- Long feedback cycles
	- Anti-pattern

# Implementation Service Tests

- Service tets
	- Deploy an instance of the customer service
	- Stub out any downstream services
- Create a binary artifact of the service
- Service test suite needs to launch stub services for any downstream collaborator
- Configure the service under tests to connect to the stub services
- Stub send responses back to mimic the real-world services

## Mocking or Stubbing

- Stubbing downstream collaborators
	- Create stub services will canned responses to known requests from the service under test
	- Called as many times
- Mock
	- Make sure call was made
	- Ensure that the expected side effect happens

## A Smarter Stub Service

- Small software appliance that is programmable via HTTP
- NodeJs
- Send it commands telling it what port to stub on, what protocol to handle, and the responses it should send when requests are send
- Supports setting expectations
- Add or remove these stub endpoints

# Those Tricky End-to-End Tests

- Deploy multiple services together
- Run a test against all of them

<img src="/images/Pasted image 20260428105625.png" alt="image" width="500">

- Multiple pipelines fan in to a single, end-to-end test stage
- When a new build of the service is triggered, run end-to-end tests

<img src="/images/Pasted image 20260428105720.png" alt="image" width="500">

# Downsides to End-to-End Testing

# Flaky and Brittle Tests

- A unused service can fail and interrupt the entire system
- Temporary network glitch could cause a test to fail without notifying the test
- Flaky tests
	- Tests that do not tell us about the error
	- Normalization of deviance

## Who Write These Tests?

- Treat the end-to-end test suite as a shared codebase, but with joint ownership

## How Long?

- 

## The Great Pile-Up?

## The Metaversion

# Test Journeys, Not Stories

## Pact


# Consumer-Driven Tests to the Rescue

## About Conversations


# So Should You Use End-to-End Tests?

## Separating Deployment from Release




# Testing After Production

## Canary Releasing

## Mean Time to Repair Over Mean Time Between Failures?


# Cross-Functional Testing

## Performance Tests

# Summary