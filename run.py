import sys
import datetime
import simple_class

try: 

    ts = sys.argv[1]

except : 
    
    ts = datetime.datetime.now()

model = simple_class.simple_mfg(str(ts))
model.warmstart()
model.train()
model.draw()
model.draw_history()
