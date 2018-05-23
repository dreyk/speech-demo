import pandas as pd
import vertica_python
import logging

def main():
    logging.getLogger().setLevel('INFO')
    logging.info('Read datas')
    data = pd.read_csv('aps1_sum_201805171505.csv')
    conn_info = {
        'host': '172.20.0.163',
        'port': 5433,
        'user': 'dbadmin',
        'password': 'vertica',
        'database': 'vertica',
        # 10 minutes timeout on queries
        'read_timeout': 600,
        # default throw error on invalid UTF-8 results
        'unicode_error': 'strict',
        # SSL is disabled by default
        'ssl': False,
        'connection_timeout': 5
        # connection timeout is not enabled by default
    }
    # simple connection, with manual close
    connection = vertica_python.connect(**conn_info)
    cur = connection.cursor()
    cur.execute('DROP TABLE aps1_sum')
    cur.execute('CREATE TABLE IF NOT EXISTS aps1_sum (value FLOAT, clock INT,itemid INT)')
    cur.close()
    cur = connection.cursor()
    values=[]
    logging.info('Start')
    for i, row in data.iterrows():
        values.append("INSERT INTO aps1_sum (value, clock,itemid) VALUES ({},{},{})".format(row['value'],row['clock'],row['itemid']))
        if i >0 and i % 200 == 0:
            cur.execute(";".join(values))
            logging.info('{}: {}%'.format(i,i*100.0/data.shape[0]))
            values=[]
    if len(values)>0:
        cur.execute(";".join(values))
    logging.info('Commit')
    connection.commit()
    connection.close()

if __name__ == '__main__':
    main()