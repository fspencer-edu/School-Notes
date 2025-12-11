
```c
SELECT d.DepartmentName, e.Name AS ManagerName
FROM Department d
LEFT JOIN Employee e ON d.ManagerID = e.EmployeeID;
```
$π Department.DepartmentName, Employee.Name (Department ⟕_{Department.ManagerID = Employee.EmployeeID}Employee)$

```c
SELECT b.Name AS BankName, br.BranchName, br.Address, br.Phone
FROM Bank b
JOIN Branch br ON b.BankID = br.BankID;
```
$πBank.Name, Branch.BranchName, Branch.Address, Branch.Phone​(BankBank.BankID=Branch.BankID⋈​Branch)$

```c
SELECT BranchID, COUNT(DISTINCT CustomerID) AS NumCustomers
FROM Customer
GROUP BY BranchID
ORDER BY NumCustomers DESC;
```
$γBranchID,COUNT(CustomerID)→NumCustomers​(Customer)$

```c
SELECT DISTINCT Name
FROM Customer
ORDER BY Name ASC;
```
$πName​(Customer)$

```c
SELECT a.AccountID, c.Name, a.AccountType, a.Balance, a.Status
FROM Account a
JOIN Customer c ON a.CustomerID = c.CustomerID;
```
$πAccount.AccountID, Customer.Name, Account.AccountType, Account.Balance, Account.Status​(AccountAccount.CustomerID=Customer.CustomerID⋈​Customer)$

```c
SELECT c.Name AS CustomerName, u.username, u.IP_Address, u.LastLogin
FROM UserLogin u
JOIN Customer c ON u.CustomerID = c.CustomerID
WHERE u.Username = 'davidlee';
```
$πCustomer.Name, UserLogin.Username, UserLogin.IP_Address, UserLogin.LastLogin​(σUserLogin.Username = ’davidlee’​(UserLoginUserLogin.CustomerID=Customer.CustomerID⋈​Customer))$

```c
SELECT DISTINCT c.CustomerID, c.Name
FROM Customer c
JOIN Account a ON c.CustomerID = a.CustomerID
WHERE a.AccountType = 'Checking'
MINUS
SELECT DISTINCT c.CustomerID, c.Name
FROM Customer c
JOIN Account a ON c.CustomerID = a.CustomerID
WHERE a.AccountType = 'Savings';
```
$CheckingSet=πCustomer.CustomerID, Customer.Name​(σAccount.AccountType = ’Checking’​(CustomerCustomer.CustomerID=Account.CustomerID⋈​Account))$
$SavingsSet=πCustomer.CustomerID, Customer.Name​(σAccount.AccountType = ’Savings’​(CustomerCustomer.CustomerID=Account.CustomerID⋈​Account))$
$Difference = CheckingSet−SavingsSet$

```c
SELECT 
    BranchName,
    NumTransactions,
    TotalTransacted
FROM View_BranchTransactions
WHERE TotalTransacted > 0
ORDER BY TotalTransacted DESC;
```
$πBranchName, NumTransactions, TotalTransacted​(σTotalTransacted > 0​(View_BranchTransactions))$

<img src="/images/Pasted image 20251211130220.png" alt="image" width="500">
