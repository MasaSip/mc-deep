from importlib import reload

import main
reload(main)
main.params[0]['skip_training'] = True
main.McDiippi('cpu',**main.params[0])

