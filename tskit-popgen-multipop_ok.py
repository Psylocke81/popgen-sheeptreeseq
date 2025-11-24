import tskit
import msprime
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# PARTE 1: Genera tree sequence con demografia complessa multi-popolazione
# =============================================================================

print("=" * 70)
print("GENERAZIONE TREE SEQUENCE - DEMOGRAFIA MULTI-POPOLAZIONE")
print("=" * 70)

# Parametri della simulazione
sequence_length = 100000  # 100kb
recombination_rate = 1e-8
mutation_rate = 1e-8
Ne = 10000

# Crea demografia complessa con 6 popolazioni moderne
demography = msprime.Demography()

# Popolazioni moderne (presenti al tempo 0)
demography.add_population(name="PopA", initial_size=Ne, description="Popolazione A")
demography.add_population(name="PopB", initial_size=Ne, description="Popolazione B")
demography.add_population(name="PopC", initial_size=Ne, description="Popolazione C")
demography.add_population(name="PopD", initial_size=Ne, description="Popolazione D")
demography.add_population(name="PopE", initial_size=Ne, description="Popolazione E")
demography.add_population(name="PopF", initial_size=Ne, description="Popolazione F")

# Popolazioni ancestrali
demography.add_population(name="AncAB", initial_size=Ne, description="Ancestrale A+B")
demography.add_population(name="AncCD", initial_size=Ne, description="Ancestrale C+D")
demography.add_population(name="AncEF", initial_size=Ne, description="Ancestrale E+F")
demography.add_population(name="AncABCD", initial_size=Ne, description="Ancestrale ABCD")
demography.add_population(name="AncAll", initial_size=Ne*2, description="Ancestrale tutte")

# Eventi demografici - split sequenziali
# Split recenti
demography.add_population_split(time=500, derived=["PopE", "PopF"], ancestral="AncEF")
demography.add_population_split(time=800, derived=["PopC", "PopD"], ancestral="AncCD")
demography.add_population_split(time=1000, derived=["PopA", "PopB"], ancestral="AncAB")

# Split intermedi
demography.add_population_split(time=2000, derived=["AncAB", "AncCD"], ancestral="AncABCD")

# Split antico - tutte le popolazioni si uniscono
demography.add_population_split(time=5000, derived=["AncABCD", "AncEF"], ancestral="AncAll")

# Aggiungi migrazione tra alcune popolazioni (opzionale)
# Migrazione simmetrica tra PopA e PopB (popolazioni sorelle)
demography.add_migration_rate_change(time=0, rate=1e-4, source="PopA", dest="PopB")
demography.add_migration_rate_change(time=0, rate=1e-4, source="PopB", dest="PopA")

# Migrazione tra PopC e PopD
demography.add_migration_rate_change(time=0, rate=5e-5, source="PopC", dest="PopD")
demography.add_migration_rate_change(time=0, rate=5e-5, source="PopD", dest="PopC")

# Ordina gli eventi per tempo (IMPORTANTE!)
demography.sort_events()

print("\nDemografia configurata:")
print(demography)

# Simula ancestry
print("\nSimulazione in corso...")
ts = msprime.sim_ancestry(
    samples={
        "PopA": 8,
        "PopB": 8,
        "PopC": 8,
        "PopD": 8,
        "PopE": 8,
        "PopF": 8
    },
    demography=demography,
    sequence_length=sequence_length,
    recombination_rate=recombination_rate,
    random_seed=42
)

# Aggiungi mutazioni
ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=42)

# Salva
ts.dump("multi_pop_tree_sequence.trees")
print("✓ Tree sequence salvato in 'multi_pop_tree_sequence.trees'")

print(f"\nTree sequence generato:")
print(f"  - Numero totale campioni: {ts.num_samples}")
print(f"  - Numero di alberi: {ts.num_trees}")
print(f"  - Lunghezza sequenza: {ts.sequence_length}")
print(f"  - Numero di mutazioni: {ts.num_mutations}")
print(f"  - Numero di popolazioni: {ts.num_populations}")

# Estrai campioni per popolazione
pop_samples = {}
pop_names = ["PopA", "PopB", "PopC", "PopD", "PopE", "PopF"]
for i, name in enumerate(pop_names):
    pop_samples[name] = ts.samples(population=i)
    print(f"  - {name}: {len(pop_samples[name])} campioni")

# =============================================================================
# PARTE 2: Calcola Fst tra tutte le coppie di popolazioni
# =============================================================================

print(f"\n" + "=" * 70)
print("CALCOLO FST TRA TUTTE LE COPPIE DI POPOLAZIONI")
print("=" * 70)

n_pops = len(pop_names)
fst_matrix = np.zeros((n_pops, n_pops))

print("\nMatrice Fst:")
for i in range(n_pops):
    for j in range(n_pops):
        if i == j:
            fst_matrix[i, j] = 0
        elif i < j:
            fst = ts.Fst(sample_sets=[pop_samples[pop_names[i]], 
                                      pop_samples[pop_names[j]]], 
                         mode="site")
            fst_matrix[i, j] = fst
            fst_matrix[j, i] = fst

# Visualizza matrice Fst
print("\n" + " " * 12 + "  ".join(pop_names))
for i, name in enumerate(pop_names):
    row_str = f"{name:>8}  "
    row_str += "  ".join([f"{fst_matrix[i, j]:.4f}" for j in range(n_pops)])
    print(row_str)

# Plot heatmap Fst
plt.figure(figsize=(10, 8))
sns.heatmap(fst_matrix, annot=True, fmt='.4f', cmap='YlOrRd', 
            xticklabels=pop_names, yticklabels=pop_names,
            cbar_kws={'label': 'Fst'})
plt.title('Matrice Fst tra popolazioni')
plt.tight_layout()
plt.savefig('fst_matrix_heatmap.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Heatmap Fst salvata in 'fst_matrix_heatmap.png'")

# =============================================================================
# PARTE 3: Calcola matrice GRM per tutti i campioni
# =============================================================================

print(f"\n" + "=" * 70)
print("CALCOLO MATRICE GRM")
print("=" * 70)

all_samples = ts.samples()
n_samples = len(all_samples)

print(f"\nCalcolo GRM per {n_samples} campioni...")
# Calcola matrice di relatedness con indici espliciti
sample_sets = [[s] for s in all_samples]
indices = [(i, j) for i in range(n_samples) for j in range(n_samples)]
grm = ts.genetic_relatedness(
    sample_sets=sample_sets,
    indexes=indices,
    mode="site",
    proportion=False
)
# Rimodella in matrice
grm = grm.reshape((n_samples, n_samples))

# Normalizza GRM
scale_factor = np.mean(np.diag(grm))
grm_normalized = grm / scale_factor if scale_factor > 0 else grm

print(f"✓ GRM calcolata")
print(f"  Shape: {grm_normalized.shape}")
print(f"  Media diagonale: {np.mean(np.diag(grm_normalized)):.4f}")

# Calcola statistiche intra/inter popolazione
print(f"\nParentela media intra- e inter-popolazione:")
for i, name_i in enumerate(pop_names):
    idx_i = [k for k, s in enumerate(all_samples) if s in pop_samples[name_i]]
    
    # Intra-popolazione
    intra = grm_normalized[np.ix_(idx_i, idx_i)]
    intra_offdiag = intra[~np.eye(len(idx_i), dtype=bool)]
    
    print(f"\n  {name_i}:")
    print(f"    Intra-{name_i}: {np.mean(intra_offdiag):.6f}")
    
    # Inter-popolazione
    for j, name_j in enumerate(pop_names):
        if j > i:
            idx_j = [k for k, s in enumerate(all_samples) if s in pop_samples[name_j]]
            inter = grm_normalized[np.ix_(idx_i, idx_j)]
            print(f"    {name_i} <-> {name_j}: {np.mean(inter):.6f}")

# Salva GRM
np.save("grm_multi_pop.npy", grm_normalized)
print(f"\n✓ GRM salvata in 'grm_multi_pop.npy'")

# =============================================================================
# PARTE 4: PCA sulla GRM
# =============================================================================

print(f"\n" + "=" * 70)
print("PCA SULLA MATRICE GRM")
print("=" * 70)

pca_grm = PCA(n_components=min(10, n_samples))
pca_grm.fit(grm_normalized)

print(f"\nVarianza spiegata dai primi 5 PC:")
for i, var in enumerate(pca_grm.explained_variance_ratio_[:5]):
    print(f"  PC{i+1}: {var*100:.2f}%")

# Trasforma i dati
pc_scores = pca_grm.transform(grm_normalized)

# Crea etichette e colori per popolazione
sample_labels = []
sample_colors = []
color_map = {
    "PopA": "red",
    "PopB": "orange", 
    "PopC": "green",
    "PopD": "lightgreen",
    "PopE": "blue",
    "PopF": "purple"
}

for sample in all_samples:
    for pop_name, samples in pop_samples.items():
        if sample in samples:
            sample_labels.append(pop_name)
            sample_colors.append(color_map[pop_name])
            break

# Plot PCA
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# PC1 vs PC2
ax1 = axes[0, 0]
for pop_name, color in color_map.items():
    mask = [label == pop_name for label in sample_labels]
    ax1.scatter(pc_scores[mask, 0], pc_scores[mask, 1], 
               c=color, label=pop_name, s=100, alpha=0.7, edgecolors='black')
ax1.set_xlabel(f'PC1 ({pca_grm.explained_variance_ratio_[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({pca_grm.explained_variance_ratio_[1]*100:.1f}%)')
ax1.set_title('PCA sulla GRM - PC1 vs PC2')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# PC1 vs PC3
ax2 = axes[0, 1]
for pop_name, color in color_map.items():
    mask = [label == pop_name for label in sample_labels]
    ax2.scatter(pc_scores[mask, 0], pc_scores[mask, 2], 
               c=color, label=pop_name, s=100, alpha=0.7, edgecolors='black')
ax2.set_xlabel(f'PC1 ({pca_grm.explained_variance_ratio_[0]*100:.1f}%)')
ax2.set_ylabel(f'PC3 ({pca_grm.explained_variance_ratio_[2]*100:.1f}%)')
ax2.set_title('PCA sulla GRM - PC1 vs PC3')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# PC2 vs PC3
ax3 = axes[1, 0]
for pop_name, color in color_map.items():
    mask = [label == pop_name for label in sample_labels]
    ax3.scatter(pc_scores[mask, 1], pc_scores[mask, 2], 
               c=color, label=pop_name, s=100, alpha=0.7, edgecolors='black')
ax3.set_xlabel(f'PC2 ({pca_grm.explained_variance_ratio_[1]*100:.1f}%)')
ax3.set_ylabel(f'PC3 ({pca_grm.explained_variance_ratio_[2]*100:.1f}%)')
ax3.set_title('PCA sulla GRM - PC2 vs PC3')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# Scree plot
ax4 = axes[1, 1]
n_show = min(10, len(pca_grm.explained_variance_ratio_))
ax4.bar(range(1, n_show+1), pca_grm.explained_variance_ratio_[:n_show])
ax4.set_xlabel('Componente Principale')
ax4.set_ylabel('Varianza Spiegata')
ax4.set_title('Scree Plot')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pca_grm_multi_pop.png', dpi=300, bbox_inches='tight')
print(f"✓ PCA sulla GRM salvata in 'pca_grm_multi_pop.png'")

# =============================================================================
# PARTE 5: Diversità genetica e statistiche per popolazione
# =============================================================================

print(f"\n" + "=" * 70)
print("STATISTICHE PER POPOLAZIONE")
print("=" * 70)

stats_dict = {
    'Popolazione': [],
    'π (diversità)': [],
    "Tajima's D": [],
    'Siti segreganti': []
}

for pop_name in pop_names:
    samples = pop_samples[pop_name]
    
    pi = ts.diversity(sample_sets=[samples], mode="site")[0]
    tajd = ts.Tajimas_D(sample_sets=[samples], mode="site")[0]
    seg_sites = ts.segregating_sites(sample_sets=[samples], mode="site")[0]
    
    stats_dict['Popolazione'].append(pop_name)
    stats_dict['π (diversità)'].append(pi)
    stats_dict["Tajima's D"].append(tajd)
    stats_dict['Siti segreganti'].append(seg_sites)
    
    print(f"\n{pop_name}:")
    print(f"  π (diversità nucleotidica): {pi:.6f}")
    print(f"  Tajima's D: {tajd:.4f}")
    print(f"  Siti segreganti: {seg_sites}")

# Plot statistiche
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Diversità nucleotidica
ax1 = axes[0]
colors_list = [color_map[pop] for pop in pop_names]
ax1.bar(pop_names, stats_dict['π (diversità)'], color=colors_list, alpha=0.7, edgecolor='black')
ax1.set_ylabel('π (diversità)')
ax1.set_title('Diversità nucleotidica per popolazione')
ax1.grid(True, alpha=0.3, axis='y')

# Tajima's D
ax2 = axes[1]
ax2.bar(pop_names, stats_dict["Tajima's D"], color=colors_list, alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax2.set_ylabel("Tajima's D")
ax2.set_title("Tajima's D per popolazione")
ax2.grid(True, alpha=0.3, axis='y')

# Siti segreganti
ax3 = axes[2]
ax3.bar(pop_names, stats_dict['Siti segreganti'], color=colors_list, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Numero di siti')
ax3.set_title('Siti segreganti per popolazione')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('pop_statistics.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Statistiche per popolazione salvate in 'pop_statistics.png'")

print(f"\n" + "=" * 70)
print("ANALISI COMPLETATA!")
print("=" * 70)
print("\nFile generati:")
print("  - multi_pop_tree_sequence.trees (tree sequence)")
print("  - grm_multi_pop.npy (matrice GRM)")
print("  - fst_matrix_heatmap.png (heatmap Fst)")
print("  - pca_grm_multi_pop.png (PCA sulla GRM)")
print("  - pop_statistics.png (statistiche per popolazione)")
