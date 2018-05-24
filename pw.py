import ConfigParser
Config = ConfigParser.ConfigParser(allow_no_value=True)
with open("test.ini",'w') as cfgfile:
    Config.add_section('PARAMETERS')
    Config.set('PARAMETERS','learning_rate',0.001)
    Config.set('PARAMETERS','mini_batch', None)
    Config.set('PARAMETERS','is_train', True)
    Config.write(cfgfile)