import argparse
import ConfigParser

def main():
    Config = ConfigParser.ConfigParser(allow_no_value=True)
    Config.read('test.ini')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='learning_rate',
    )
    parser.add_argument(
        '--mini_batch',
        type=int,
        default=1,
        help='mini_batch',
    )
    parser.add_argument(
        '--is_train',
        action='store_true',
        help='is_train',
    )
    parser.add_argument(
        '--other',
        default='Test'
    )
    parser.set_defaults(**dict(Config.items("PARAMETERS")))
    args = parser.parse_args()
    print('{} - {} - {} - {}'.format(args.learning_rate,args.mini_batch,args.is_train,args.other))

if __name__ == '__main__':
    main()
