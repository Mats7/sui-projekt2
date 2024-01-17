# %% [markdown]
# Vítejte u druhého projektu do SUI.
# V rámci projektu Vás čeká několik cvičení, v nichž budete doplňovat poměrně malé fragmenty kódu (místo je vyznačeno pomocí `None` nebo `pass`).
# Pokud se v buňce s kódem již něco nachází, využijte/neničte to.
# Buňky nerušte ani nepřidávejte.
# Snažte se programovat hezky, ale jediná skutečně aktivně zakázaná, vyhledávaná a -- i opakovaně -- postihovaná technika je cyklení přes data (ať už explicitním cyklem nebo v rámci `list`/`dict` comprehension), tomu se vyhýbejte jako čert kříží a řešte to pomocí vhodných operací lineární algebry.
# 
# Až budete s řešením hotovi, vyexportujte ho ("Download as") jako PDF i pythonovský skript a ty odevzdejte pojmenované názvem týmu (tj. loginem vedoucího).
# Dbejte, aby bylo v PDF všechno vidět (nezůstal kód za okrajem stránky apod.).
# 
# U všech cvičení je uveden orientační počet řádků řešení.
# Berte ho prosím opravdu jako orientační, pozornost mu věnujte, pouze pokud ho významně překračujete.

# %%
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.stats

# %% [markdown]
# # Přípravné práce
# Prvním úkolem v tomto projektu je načíst data, s nimiž budete pracovat.
# Vybudujte jednoduchou třídu, která se umí zkonstruovat z cesty k negativním a pozitivním příkladům, a bude poskytovat:
# - pozitivní a negativní příklady (`dataset.pos`, `dataset.neg` o rozměrech [N, 7])
# - všechny příklady a odpovídající třídy (`dataset.xs` o rozměru [N, 7], `dataset.targets` o rozměru [N])
# 
# K načítání dat doporučujeme využít `np.loadtxt()`.
# Netrapte se se zapouzdřováním a gettery, berte třídu jako Plain Old Data.
# 
# Načtěte trénovací (`{positives,negatives}.trn`), validační (`{positives,negatives}.val`) a testovací (`{positives,negatives}.tst`) dataset, pojmenujte je po řadě `train_dataset`, `val_dataset` a `test_dataset`. 
# 
# **(6 řádků)** 
# 

# %%
class BinaryDataset:
    def __init__(self, d1, d2):
        self.pos = np.loadtxt(d1)
        self.neg = np.loadtxt(d2)
        self.xs = np.append(self.pos, self.neg, axis=0)
        self.targets = np.concatenate((np.ones(self.pos.shape[0]), np.zeros(self.neg.shape[0])))

train_dataset = BinaryDataset('positives.trn', 'negatives.trn')
val_dataset = BinaryDataset('positives.val', 'negatives.val')
test_dataset = BinaryDataset('positives.tst', 'negatives.tst')

print('positives', train_dataset.pos.shape)
print('negatives', train_dataset.neg.shape)
print('xs', train_dataset.xs.shape)
print('targets', train_dataset.targets.shape)

# %% [markdown]
# V řadě následujících cvičení budete pracovat s jedním konkrétním příznakem. Naimplementujte proto funkci, která vykreslí histogram rozložení pozitivních a negativních příkladů z jedné sady. Nezapomeňte na legendu, ať je v grafu jasné, které jsou které. Funkci zavoláte dvakrát, vykreslete histogram příznaku `5` -- tzn. šestého ze sedmi -- pro trénovací a validační data
# 
# **(5 řádků)**

# %%
FOI = 5  # Feature Of Interest

def plot_data(poss, negs):
    plt.hist(poss, orientation='vertical', bins=200, color='green', label='poss', alpha=0.5)
    plt.hist(negs, orientation='vertical', bins=200, color='red', label='negs', alpha=0.5)
    plt.legend()
    plt.show()

plot_data(train_dataset.pos[:, FOI], train_dataset.neg[:, FOI])
plot_data(val_dataset.pos[:, FOI], val_dataset.neg[:, FOI])

# %% [markdown]
# ### Evaluace klasifikátorů
# Než přistoupíte k tvorbě jednotlivých klasifikátorů, vytvořte funkci pro jejich vyhodnocování.
# Nechť se jmenuje `evaluate` a přijímá po řadě klasifikátor, pole dat (o rozměrech [N, F]) a pole tříd ([N]).
# Jejím výstupem bude _přesnost_ (accuracy), tzn. podíl správně klasifikovaných příkladů.
# 
# Předpokládejte, že klasifikátor poskytuje metodu `.prob_class_1(data)`, která vrací pole posteriorních pravděpodobností třídy 1 pro daná data.
# Evaluační funkce bude muset provést tvrdé prahování (na hodnotě 0.5) těchto pravděpodobností a srovnání získaných rozhodnutí s referenčními třídami.
# Využijte fakt, že `numpy`ovská pole lze mj. porovnávat se skalárem.
# 
# **(3 řádky)**

# %%
def evaluate(classifier, inputs, targets):
    classified = (classifier.prob_class_1(inputs) > 0.5)
    match_ratio = np.sum(classified.astype(int) == targets) / targets.size
    return match_ratio

class Dummy:
    def prob_class_1(self, xs):
        return np.asarray([0.2, 0.7, 0.7])

print(evaluate(Dummy(), None, np.asarray([0, 0, 1])))  # should be 0.66

# %% [markdown]
# ### Baseline
# Vytvořte klasifikátor, který ignoruje vstupní data.
# Jenom v konstruktoru dostane třídu, kterou má dávat jako tip pro libovolný vstup.
# Nezapomeňte, že jeho metoda `.prob_class_1(data)` musí vracet pole správné velikosti.
# 
# **(4 řádky)**

# %%
class PriorClassifier:
    def __init__(self, class_tip):
        self.class_tip = class_tip
    
    def prob_class_1(self, data):
        return np.full(shape=val_dataset.targets.shape, fill_value=self.class_tip)       

baseline = PriorClassifier(0)
val_acc = evaluate(baseline, val_dataset.xs[:, FOI], val_dataset.targets)
print('Baseline val acc:', val_acc)

# %% [markdown]
# # Generativní klasifikátory
# V této  části vytvoříte dva generativní klasifikátory, oba založené na Gaussovu rozložení pravděpodobnosti.
# 
# Začněte implementací funce, která pro daná 1-D data vrátí Maximum Likelihood odhad střední hodnoty a směrodatné odchylky Gaussova rozložení, které data modeluje.
# Funkci využijte pro natrénovaní dvou modelů: pozitivních a negativních příkladů.
# Získané parametry -- tzn. střední hodnoty a směrodatné odchylky -- vypíšete.
# 
# **(1 řádek)**

# %%
def mle_gauss_1d(data):
    return np.mean(data), np.std(data, ddof=1)

mu_pos, std_pos = mle_gauss_1d(train_dataset.pos[:, FOI])
mu_neg, std_neg = mle_gauss_1d(train_dataset.neg[:, FOI])

print('Pos mean: {:.2f} std: {:.2f}'.format(mu_pos, std_pos))
print('Neg mean: {:.2f} std: {:.2f}'.format(mu_neg, std_neg))

# %% [markdown]
# Ze získaných parametrů vytvořte `scipy`ovská gaussovská rozložení `scipy.stats.norm`.
# S využitím jejich metody `.pdf()` vytvořte graf, v němž srovnáte skutečné a modelové rozložení pozitivních a negativních příkladů.
# Rozsah x-ové osy volte od -0.5 do 1.5 (využijte `np.linspace`) a u volání `plt.hist()` nezapomeňte nastavit `density=True`, aby byl histogram normalizovaný a dal se srovnávat s modelem.
# 
# **(2 + 8 řádků)**

# %%
pos_dist = scipy.stats.norm(loc=mu_pos, scale=std_pos)
neg_dist = scipy.stats.norm(loc=mu_neg, scale=std_neg)

pos_samples = train_dataset.pos[:, FOI]
neg_samples = train_dataset.neg[:, FOI]
plt.hist(pos_samples, density=True, bins=20, color='deepskyblue', label='pos', alpha=0.5)
plt.hist(neg_samples, density=True, bins=20, color='magenta', label='neg', alpha=0.5)

x_pdf = np.linspace(-0.5, 1.5, train_dataset.pos.shape[0])
pos_pdf = pos_dist.pdf(x_pdf)
neg_pdf = neg_dist.pdf(x_pdf)
plt.plot(x_pdf, pos_pdf, 'blue', linewidth=3, label="pos_pdf")
plt.plot(x_pdf, neg_pdf, 'red', linewidth=3, label="neg_pdf")

plt.legend()
plt.show()

# %% [markdown]
# Naimplementujte binární generativní klasifikátor. 
# Při konstrukci přijímá dvě rozložení poskytující metodu `.pdf()` a odpovídající apriorní pravděpodobnost tříd.
# Dbejte, aby Vám uživatel nemohl zadat neplatné apriorní pravděpodobnosti.
# Jako všechny klasifikátory v tomto projektu poskytuje metodu `prob_class_1()`.
# 
# **(9 řádků)**

# %%
class GenerativeClassifier2Class:
    def __init__(self, dist1, dist2, prior_val):
        self.dist1 = dist1
        self.dist2 = dist2
        self.prior_val = prior_val
        if prior_val.any() > 1 or prior_val.any() < 0:
            raise ValueError("Incorrect aprior...")
    
    def prob_class_1(self, xs):
        return self.prior_val[1] * self.dist1.pdf(xs) / (self.prior_val[1] * self.dist1.pdf(xs) + self.prior_val[0] * self.dist2.pdf(xs))

# %% [markdown]
# Nainstancujte dva generativní klasifikátory: jeden s rovnoměrnými priory a jeden s apriorní pravděpodobností 0.75 pro třídu 0 (negativní příklady).
# Pomocí funkce `evaluate()` vyhodnotíte jejich úspěšnost na validačních datech.
# 
# **(2 řádky)**

# %%
classifier_flat_prior = GenerativeClassifier2Class(pos_dist, neg_dist, np.array([0.5, 0.5]))
classifier_full_prior = GenerativeClassifier2Class(pos_dist, neg_dist, np.array([0.75, 0.25]))

print('flat:', evaluate(classifier_flat_prior, val_dataset.xs[:, FOI], val_dataset.targets))
print('full:', evaluate(classifier_full_prior, val_dataset.xs[:, FOI], val_dataset.targets))

# %% [markdown]
# Vykreslete průběh posteriorní pravděpodobnosti třídy 1 jako funkci příznaku 5, opět v rozsahu <-0.5; 1.5> pro oba klasifikátory.
# Do grafu zakreslete i histogramy rozložení trénovacích dat, opět s `density=True` pro zachování dynamického rozsahu.
# 
# **(8 řádků)**

# %%
pos_samples = train_dataset.pos[:, FOI]
neg_samples = train_dataset.neg[:, FOI]
plt.hist(pos_samples, density=True, bins=20, color='deepskyblue', label='pos', alpha=0.5)
plt.hist(neg_samples, density=True, bins=20, color='magenta', label='neg', alpha=0.5)

x = np.linspace(-0.5, 1.5, train_dataset.pos.shape[0])
plt.plot(x, classifier_flat_prior.prob_class_1(x), 'blue', linewidth=3, label="flat")
plt.plot(x, classifier_full_prior.prob_class_1(x), 'red', linewidth=3, label="full")

plt.legend()
plt.show()

# %% [markdown]
# # Diskriminativní klasifikátory
# V následující části budete pomocí (lineární) logistické regrese přímo modelovat posteriorní pravděpodobnost třídy 1.
# Modely budou založeny čistě na NumPy, takže nemusíte instalovat nic dalšího.
# Nabitějších toolkitů se dočkáte ve třetím projektu.

# %%
def logistic_sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def binary_cross_entropy(probs, targets):
    return np.sum(-targets * np.log(probs) - (1-targets)*np.log(1-probs)) 

class LogisticRegressionNumpy:
    def __init__(self, dim):
        self.w = np.array([0.0] * dim)
        self.b = np.array([0.0])
        
    def prob_class_1(self, x):
        return logistic_sigmoid(x @ self.w + self.b)

# %% [markdown]
# Diskriminativní klasifikátor očekává, že dostane vstup ve tvaru `[N, F]`.
# Pro práci na jediném příznaku bude tedy zapotřebí vyřezávat příslušná data v správném formátu (`[N, 1]`). 
# Doimplementujte třídu `FeatureCutter` tak, aby to zařizovalo volání její instance.
# Který příznak se použije, nechť je konfigurováno při konstrukci.
# 
# Může se Vám hodit `np.newaxis`.
# 
# **(2 řádky)**

# %%
class FeatureCutter:
    def __init__(self, fea_id):
        self.fea_id = fea_id
        
    def __call__(self, x : np.ndarray) -> np.ndarray:
        return x[:, self.fea_id, np.newaxis]

# %% [markdown]
# Dalším krokem je implementovat funkci, která model vytvoří a natrénuje.
# Jejím výstupem bude (1) natrénovaný model, (2) průběh trénovací loss a (3) průběh validační přesnosti.
# Neuvažujte žádné minibatche, aktualizujte váhy vždy na celém trénovacím datasetu.
# Po každém kroku vyhodnoťte model na validačních datech.
# Jako model vracejte ten, který dosáhne nejlepší validační přesnosti.
# Jako loss použijte binární cross-entropii  a logujte průměr na vzorek.
# Pro výpočet validační přesnosti využijte funkci `evaluate()`.
# Oba průběhy vracejte jako obyčejné seznamy.
# 
# **(cca 11 řádků)**

# %%
def train_logistic_regression(nb_epochs, lr, in_dim, fea_preprocessor):
    model = LogisticRegressionNumpy(in_dim)
    best_model : LogisticRegressionNumpy = copy.deepcopy(model)
    losses = []
    accuracies = []
    
    train_X = fea_preprocessor(train_dataset.xs)
    train_t = train_dataset.targets

    # validation DS
    val_X = fea_preprocessor(val_dataset.xs)
    val_t = val_dataset.targets

    n = train_X.shape[0]

    max_accuracy = 0.0

    for epoch in range(nb_epochs):

        probs = model.prob_class_1(train_X)

        dw = np.dot(train_X.T, (probs - train_t))
        db = np.sum(probs - train_t)

        model.w -= lr * dw
        model.b -= lr * db

        losses.append(binary_cross_entropy(probs, train_t) / n)

        accuracy = evaluate(model, val_X, val_t)
        accuracies.append(accuracy)
        
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_model = copy.deepcopy(model)

        
    return best_model, losses, accuracies


# %% [markdown]
# Funkci zavolejte a natrénujte model.
# Uveďte zde parametry, které vám dají slušný výsledek.
# Měli byste dostat přesnost srovnatelnou s generativním klasifikátorem s nastavenými priory.
# Neměli byste potřebovat víc, než 100 epoch.
# Vykreslete průběh trénovací loss a validační přesnosti, osu x značte v epochách.
# 
# V druhém grafu vykreslete histogramy trénovacích dat a pravděpodobnost třídy 1 pro x od -0.5 do 1.5, podobně jako výše u generativních klasifikátorů.
# 
# **(1 + 5 + 8 řádků)**

# %%
disc_fea5, losses, accuracies = train_logistic_regression(nb_epochs=100, lr=0.0003, in_dim=1, fea_preprocessor= FeatureCutter(FOI))

plt.plot(range(1, len(losses) + 1), losses, label='Training Loss', color='blue')
plt.plot(range(1, len(accuracies) + 1), accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.title('Training Loss and Validation Accuracy Over Epochs')
plt.legend()
plt.show()

print('w', disc_fea5.w.item(), 'b', disc_fea5.b.item())

pos_samples = train_dataset.pos[:, FOI]
neg_samples = train_dataset.neg[:, FOI]
plt.hist(pos_samples, density=True, bins=20, color='deepskyblue', label='pos', alpha=0.5)
plt.hist(neg_samples, density=True, bins=20, color='magenta', label='neg', alpha=0.5)
x = np.linspace(-0.5, 1.5, train_dataset.pos.shape[0])
plt.plot(x, disc_fea5.prob_class_1(x[:, np.newaxis]), 'red', linewidth=3, label="disc_fea5")
plt.legend()
plt.show()

print('disc_fea5:', evaluate(disc_fea5, val_dataset.xs[:, FOI][:, np.newaxis], val_dataset.targets))

# %% [markdown]
# ## Všechny vstupní příznaky
# V posledním cvičení natrénujete logistickou regresi, která využije všechn sedm vstupních příznaků.
# Zavolejte funkci z předchozího cvičení, opět vykreslete průběh trénovací loss a validační přesnosti.
# Měli byste se dostat nad 90 % přesnosti.
# 
# Může se Vám hodit `lambda` funkce.
# 
# **(1 + 5 řádků)**

# %%
disc_full_fea, losses, accuracies = train_logistic_regression(nb_epochs=250, lr=0.0000004, in_dim=7, fea_preprocessor=lambda x: x)

print("Maximum validation accuracy: " + str(max(accuracies)))

plt.plot(range(1, len(losses) + 1), losses, label='Training Loss', color='blue')
plt.plot(range(1, len(accuracies) + 1), accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.title('Training Loss and Validation Accuracy Over Epochs')
plt.legend()
plt.show()


# %% [markdown]
# # Závěrem
# Konečně vyhodnoťte všech pět vytvořených klasifikátorů na testovacích datech.
# Stačí doplnit jejich názvy a předat jim odpovídající příznaky.
# Nezapomeňte, že u logistické regrese musíte zopakovat formátovací krok z `FeatureCutter`u.

# %%
xs_full = test_dataset.xs
xs_foi = test_dataset.xs[:, FOI]
targets = test_dataset.targets

print('Baseline:', evaluate(baseline, xs_full, targets))
print('Generative classifier (w/o prior):', evaluate(classifier_flat_prior, xs_foi, targets))
print('Generative classifier (correct):', evaluate(classifier_full_prior, xs_foi, targets))
print('Logistic regression:', evaluate(disc_fea5, xs_foi[:, np.newaxis], targets))
print('logistic regression all features:', evaluate(disc_full_fea, xs_full, targets))

# %% [markdown]
# Blahopřejeme ke zvládnutí projektu! Nezapomeňte spustit celý notebook načisto (Kernel -> Restart & Run all) a zkontrolovat, že všechny výpočty prošly podle očekávání.
# 
# Mimochodem, vstupní data nejsou synteticky generovaná.
# Nasbírali jsme je z baseline řešení historicky prvního SUI projektu; vaše klasifikátory v tomto projektu predikují, že daný hráč vyhraje dicewars, takže by se daly použít jako heuristika pro ohodnocování listových uzlů ve stavovém prostoru hry.
# Pro představu, data jsou z pozic pět kol před koncem partie pro daného hráče.
# Poskytnuté příznaky popisují globální charakteristiky stavu hry jako je například poměr délky hranic předmětného hráče k ostatním hranicím.
# Nejeden projekt v ročníku 2020 realizoval požadované "strojové učení" kopií domácí úlohy.

# %%



