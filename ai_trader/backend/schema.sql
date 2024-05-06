CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_email TEXT UNIQUE NOT NULL,
  risk_level TEXT
);

CREATE TABLE performance_old (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date DATETIME NOT NULL,
  account_value FLOAT, 
  aa FLOAT,
  ensemble FLOAT,
  dji FLOAT, 
  minvar float,
  tangent float
);

CREATE TABLE performance (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date DATETIME NOT NULL,
  account_value FLOAT, 
);

CREATE TABLE stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE
);

CREATE TABLE stockPercentages (
    id INT PRIMARY KEY AUTOINCREMENT,
    date DATE,
    stockID INT,
    percentage FLOAT,
    FOREIGN KEY (StockID) REFERENCES Stocks(ID)
);