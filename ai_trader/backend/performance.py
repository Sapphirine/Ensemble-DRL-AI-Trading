from db import get_db

class Performance:
    def __init__(self, df):
        self.id = df['id']
        self.date=df['date']
        self.account_value = df['account_value']
        self.aa = df['aa']
        self.ensemble = df['ensemble']
        self.dji = df['dji']
        self.minvar = df['minvar']
        self.tangent = df['tangent']

    @staticmethod
    def get(perf_id):
        db = get_db()
        row = db.execute(
            "SELECT * FROM ratings WHERE id = ?", (perf_id)
        ).fetchone()
        if not row:
            return None

        perf = Performance(
                        id=row[0],
                        date=row[1],
                        account_value=row[2],
                        aa=row[3],
                        ensemble=row[4],
                        dji=row[5],
                        minvar=row[6], 
                        tangent=row[7]
                       )
        return perf

    @staticmethod
    def create(date, account_value, aa, ensemble, dji, minvar, tangent):

        db = get_db()
        db.execute(
            "INSERT INTO performance (date, ticker, stock_price, stock_qty, balance)"
            "VALUES (?, ?, ?, ?, ?)", (date, account_value, aa, ensemble, dji, minvar, tangent),
        )
        db.commit()
