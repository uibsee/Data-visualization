# distribution plots
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import seaborn as sns
sns.set()
font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font_name)
df=pd.read_excel('2018test.xlsx',sheet_name='test')


math=df.수학1.values
sns.distplot(math, kde=True, rug=True )

plt.show()



# Relating variables with scatter plots
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import seaborn as sns
sns.set()
font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font_name)
df=pd.read_excel('2018test2.xlsx',sheet_name='test')

sns.relplot(x="수학1", y="생명과학", hue="반",style='반', palette='Paired', data=df);

plt.show()




# Linear regression with marginal distributions
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import seaborn as sns
sns.set()
font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font_name)
df=pd.read_excel('2018test.xlsx',sheet_name='test')

g = sns.jointplot("수학1", "수학2", data=df, kind="reg",
                  xlim=(0, 100), ylim=(0, 100), color="m", height=7)

plt.show()

# Linear regression with marginal distributions2
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import seaborn as sns
sns.set()
font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font_name)
df=pd.read_excel('2018test.xlsx',sheet_name='test')

g = sns.jointplot("수학1", "수학2", data=df, kind="kde",
                  xlim=(0, 100), ylim=(0, 100), color="m", height=7)

plt.show()




# Violinplots
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)
font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font_name)
df=pd.read_excel('2018test2.xlsx',sheet_name='test')

sns.catplot(x="반", y="수학1", kind="violin", data=df);

plt.show()

# Showing multiple relationships with Violinplots
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)
font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font_name)
df=pd.read_excel('2018test2.xlsx',sheet_name='test')

df = pd.melt(df, "반", var_name="과목")
sns.catplot(x="반", y="value", col='과목', kind="violin",ylim=(0, 100), data=df);

plt.show()



# Multiple linear regression
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import seaborn as sns
sns.set()
font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font_name)
df=pd.read_excel('2018test.xlsx',sheet_name='test')

g = sns.lmplot(x="수학1", y="수학2", hue="반", truncate=True, height=5, data=df)
g.set_axis_labels("수학1", "수학2")

plt.show()





# Scatterplot with categorical variables
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid", palette="muted")
font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font_name)
df=pd.read_excel('2018test2.xlsx',sheet_name='test')

df = pd.melt(df, "반", var_name="과목")
sns.swarmplot(x="과목", y="value", hue="반", data=df)

plt.show()


# Scatterplot Matrix
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import pandas as pd
import seaborn as sns
sns.set()
font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font_name)
df=pd.read_excel('2018test2.xlsx',sheet_name='test')
sns.pairplot(df, hue='반')
plt.show()
