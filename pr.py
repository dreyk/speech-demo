import argparse
import configparser

def main():
    Config = configparser.ConfigParser(allow_no_value=True)
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
    print(Config.items("PARAMETERS",raw=True))
    parser.set_defaults(**dict(Config.items("PARAMETERS",raw=True)))
    args = parser.parse_args()
    print('{} - {} - {} - {}'.format(args.learning_rate,args.mini_batch,'train' if args.is_train else 'test',args.other))
    print(args)

if __name__ == '__main__':
    main()
