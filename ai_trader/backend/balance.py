from db import get_db

class Balance:
    def __init__(self, df):
        self.id = df['id']
        self.date=df['date']
        self.ticker = df['ticker']
        self.stock_price = df['stock_price']
        self.stock_qty = df['stock_qty']
        self.balance = df['balance']

    @staticmethod
    def get(balance_id):
        db = get_db()
        row = db.execute(
            "SELECT * FROM ratings WHERE id = ?", (balance_id)
        ).fetchone()
        if not row:
            return None

        bal = Balance(
                        id=row[0],
                        date=row[1],
                        ticker=row[2],
                        stock_price=row[3],
                        stock_qty=row[4],
                        balance=row[5],
                       )
        return bal

    @staticmethod
    def create(date, ticker, stock_price, stock_qty, balance):

        db = get_db()
        db.execute(
            "INSERT INTO balance (date, ticker, stock_price, stock_qty, balance)"
            "VALUES (?, ?, ?, ?, ?)", (date, ticker, stock_price, stock_qty, balance),
        )
        db.commit()
