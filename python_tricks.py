# a collection of handy python snippets that might be handy in the future

# dates
today = datetime.date.today()
today_string = today.strftime('%Y-%m-%d')
day = datetime.timedelta(days=1)
week = datetime.timedelta(days=7)
month = datetime.timedelta(days=30)
yesterday = today - day
yesterday_string = yesterday.strftime('%Y-%m-%d')
week_ago_string = (today - week).strftime('%Y-%m-%d')
month_ago_string = (today - month).strftime('%Y-%m-%d')


# make import statement look in other places
import sys
sys.path.append('/home/jmonello/path_to_new_directory')


# my standard matplotlib chart
fig = plt.figure(figsize = (12,7))
ax = fig.gca()
ax.plot(x_axis, y_axis, 'bo-', label = 'label')
ax.plot(another)
ax2 = ax.twinx() # for secondary axis
ax2.plot(blah)
ax.set_xlabel()
ax.set_xticklabels()
ax.legend()


# groupby magic - can have things other than sum
df.groupby('end_day').agg({'reporting_imps': 'sum', 'lifetime_budget_imps': 'sum'})
# another groupby thing, to make a cool stackplot
def ratios(group):
    reasons = competition_info.values() #competition info a dict
    count = float(group['campaign_group_id'].count())
    return pd.Series({reason: ((group['competition_label'] == reason).sum()/count) for reason in reasons})

daily_ratios = df.groupby('end_day').apply(ratios)
lines = ax.stackplot(daily_ratios_plot.index, daily_ratios_plot[cols].values.T,
                    labels=daily_ratios_plot[cols].columns, alpha=alpha)


# better way to assign new columns to dataframe
df = df.assign(new_column = logic).astype(int) #astype(int) to make it a binary flag

################################################################
# ML stuff
################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import svm
from sklearn import grid_search
from scipy import stats

# test/train split for ML
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

# precision-recall curve plot
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
def pr_curve(x,y,model_name):
    # Precision-Recall curve
    Y_score = model_name.predict_proba(x)[:,1]

    precision, recall, _ = precision_recall_curve(y[:], Y_score[:])
    average_precision = average_precision_score(y[:], Y_score[:])

    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure(figsize = (12,6))
    plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % average_precision)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', size = 'x-large')
    plt.ylabel('Precision', size = 'x-large')
    plt.title('Precision-Recall Curve', size = 'xx-large')
    plt.legend(loc="upper right")
    plt.show()

# example decision tree + showing the output
tmodel = DecisionTreeClassifier(max_depth = 3, criterion='entropy').fit(X_train, Y_train)

pr_curve(X_test, Y_test, tmodel)

from IPython.display import Image
export_graphviz(tmodel, out_file = 'trees/depth3_entropy_custom.dot')
!dot -Tpng trees/depth3_entropy_custom.dot -o trees/depth3_entropy_custom.png
Image('trees/depth3_entropy_custom.png')



###################################################
# command line arguments for python scripts
###################################################
def process_command_line():
    """Parse command line args."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Boilerplate, some useful things at times.
    parser.add_argument('--doit', help="Really do it.", action="store_true")
    parser.add_argument('--test', help="Test.", action="store_true")
    parser.add_argument('--verbose', help="Like me.", action="store_true")

    # Script specific.
    parser.add_argument('--revisionist', help="Run for day other than today.",
                        action="store_true")

    parser.add_argument('--now', help="String of 'now()' for revisionist.",
                        action="store", default='1899-03-14 12:34')

    args = parser.parse_args()

    if args.revisionist:
        assert pd.Timestamp(args.now) > pd.Timestamp("2016-09-20")  # Or similar.

    return args

args = process_command_line()
# Or change your code to use 'if args.test' instead of 'if TEST'.
TEST = args.test
logger.debug("Working, test == %s" % str(TEST))

# then in code i can do (or revisionist)
IF NOT TEST:
    blah
ELSE:
    blah blah



###################################################
# the if name == __main__ trick is good for testing stuff
###################################################
def main():
    the stuff that would run goes here (will need other functions imported cuz this is the only function)

if __name__ == '__main__':
    main()


###################################################
# ipshell - stops a script where i put ipshell() and opens in ipython shell for testing stuff
# this whole thing would be saved as ipshell.py
###################################################
# http://ipython.readthedocs.io/en/stable/interactive/reference.html#embedding
"""Quick code snippets for embedding IPython into other programs.

See embed_class_long.py for full details, this file has the bare minimum code for
cut and paste use once you understand how to use the system."""

#---------------------------------------------------------------------------
# This code loads IPython but modifies a few things if it detects it's running
# embedded in another IPython session (helps avoid confusion)

try:
    get_ipython
except NameError:
    banner=exit_msg=''
else:
    banner = '*** Nested interpreter ***'
    exit_msg = '*** Back in main IPython ***'

# First import the embed function
from IPython.terminal.embed import InteractiveShellEmbed
# Now create the IPython shell instance. Put ipshell() anywhere in your code
# where you want it to open.
ipshell = InteractiveShellEmbed(banner1=banner, exit_msg=exit_msg)

#---------------------------------------------------------------------------
# This code will load an embeddable IPython shell always with no changes for
# nested embededings.

from IPython import embed
# Now embed() will open IPython anywhere in the code.

#---------------------------------------------------------------------------
# This code loads an embeddable shell only if NOT running inside
# IPython. Inside IPython, the embeddable shell variable ipshell is just a
# dummy function.

try:
    get_ipython
except NameError:
    from IPython.terminal.embed import InteractiveShellEmbed
    ipshell = InteractiveShellEmbed()
    # Now ipshell() will open IPython anywhere in the code
else:
    # Define a dummy ipshell() so the same code doesn't crash inside an
    # interactive IPython
    def ipshell(): pass



#############################################
# PySpark stuff
#############################################

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import HiveContext
from pyspark import StorageLevel

conf = SparkConf().setMaster("yarn-client")
conf.setAppName("application_name")
conf.set('spark.yarn.queue', 'queue.location')
conf.set("spark.executor.instances", "500")
conf.set("spark.executor.memory", "4g")

sc = SparkContext(conf=conf)
hc = HiveContext(sc)

df = hc.sql(query)
#df.persist(storageLevel)
df.registerTempTable("df") # so it can be queried with another spark query
df_base_with_size.count() # to actually do the query - since it's lazy eval

# timer (for diagnostics)
class Timer:
    def __init__(self):        self.start = time.time()
    def set(self): self.start = time.time()
    def get(self):
        delta = time.time() - self.start
        self.start = time.time()
        return 'elapsed time: %f seconds.' % delta

timer = Timer()
timer_all = Timer()
