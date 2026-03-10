import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from itertools import combinations

DATA_FOLDER   = 'Sentence Memorability/NewLogsAnonymized'
OUTPUT_FOLDER = 'output'   

warnings.filterwarnings('ignore')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

COLORS = {
    'HH': '#2E86AB',
    'HL': '#A23B72',
    'LH': '#F18F01',
    'LL': '#C73E1D',
    'Active':  '#3A86FF',
    'Passive': '#FF6B6B',
    'accent':  '#264653',
}
COND_LABELS = {
    'HH': 'HH\n(Hi–Hi)',
    'HL': 'HL\n(Hi–Lo)',
    'LH': 'LH\n(Lo–Hi)',
    'LL': 'LL\n(Lo–Lo)',
}
CONDITIONS = ['HH', 'HL', 'LH', 'LL']

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.labelsize':    11,
    'axes.titlesize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'figure.dpi':        150,
})

def savefig(name):
    path = os.path.join(OUTPUT_FOLDER, name)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# SECTION 1 – PARSING & DECODING

def decode_stimulus(stim):
    """
    Parse a stimulus code like 'HL_114_A' into:
      condition  : canonical 2*2 condition ('HH','HL','LH','LL') or 'FILLER'
      sentence_id: string ID within the condition pool
      voice      : 'Active' or 'Passive'

    Stimulus naming convention observed in the log files:
      Prefix  Canonical   Meaning
      ------  ---------   -------
      HH      HH          High-subject, High-object
      HVH     HH          same (verb placeholder)
      HL      HL          High-subject, Low-object
      HVL     HL          same
      LH      LH          Low-subject, High-object
      LVH     LH          same
      LL      LL          Low-subject, Low-object
      LVL     LL          same
      HF / LF FILLER      Filler sentences (not targets)
    Suffix: _A = Active voice, _P = Passive voice
    """
    if pd.isna(stim) or str(stim).strip() in ('N/A', '', 'nan'):
        return None, None, None
    parts = str(stim).strip().split('_')
    if len(parts) < 3:
        return None, None, None
    raw  = parts[0].upper()
    sid  = parts[1]
    voice = 'Active' if parts[2].upper() == 'A' else 'Passive'
    cond_map = {
        'HH': 'HH', 'HVH': 'HH',
        'HL': 'HL', 'HVL': 'HL',
        'LH': 'LH', 'LVH': 'LH',
        'LL': 'LL', 'LVL': 'LL',
        'HF': 'FILLER', 'LF': 'FILLER',
    }
    return cond_map.get(raw, 'OTHER'), sid, voice


def parse_single_log(filepath):
    """
    Load one participant's .log file and add decoded columns.
    Returns the full raw DataFrame with extra columns:
      Condition, SentenceID, Voice, is_practice
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Participant ID from filename as fallback if column is inconsistent
    pid = os.path.splitext(os.path.basename(filepath))[0]
    if 'participant_ID' not in df.columns:
        df['participant_ID'] = pid

    # Decode stimulus
    decoded = df['Stimulus'].apply(lambda x: pd.Series(decode_stimulus(x),
                                                        index=['Condition','SentenceID','Voice']))
    df = pd.concat([df, decoded], axis=1)

    # Flag practice rows
    df['is_practice'] = df['Event'].str.contains('Practice', na=False)

    return df


def load_all_logs(folder):
    """
    Load all .log files in *folder*.
    Returns one concatenated DataFrame.
    """
    files = sorted(glob.glob(os.path.join(folder, '*.log')))
    if not files:
        raise FileNotFoundError(f"No .log files found in: {folder}")
    print(f"Found {len(files)} log files.")
    parts = []
    for f in files:
        try:
            parts.append(parse_single_log(f))
        except Exception as e:
            print(f"  WARNING: could not parse {f}: {e}")
    df = pd.concat(parts, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    return df


# SECTION 2 – BLOCK ASSIGNMENT

def assign_blocks(df_main):
    """
    Assign a block number (1, 2, 3) to every main-experiment row based on
    'Rest Phase started' events, which separate blocks in the log.

    Rows before the first rest = Block 1
    Rows between rest 1 and rest 2 = Block 2
    Rows after rest 2 = Block 3
    """
    df = df_main.copy().reset_index(drop=True)
    block = 1
    block_col = []
    for _, row in df.iterrows():
        if 'Rest Phase' in str(row['Event']):
            block += 1
        block_col.append(block)
    df['Block'] = block_col
    return df


# SECTION 3 – VALIDATION EXCLUSION (per participant, per block)

def validate_block(block_df):
    """
    Apply the validation formula to one block's rows:
      Correct_Val > (Wrong_Val / 2) + Missed_Val
    Returns True (block passes) or False (block fails → exclude).
    """
    val = block_df[block_df['isValidation'] == True]
    correct = val['Event'].str.contains('Validation IR pressed', na=False).sum()
    wrong   = val['Event'].str.contains('Validation Wrong IR pressed', na=False).sum()
    missed  = val['Event'].str.contains('Validation Missed', na=False).sum()
    threshold = (wrong / 2) + missed
    return correct > threshold, correct, wrong, missed


def apply_validation_exclusion(df_main):
    """
    For each participant × block, check the validation criterion.
    Rows from blocks that FAIL are removed from the analysis.

    Returns:
      df_valid     : rows that survive exclusion
      exclusion_log: DataFrame recording pass/fail per participant × block
    """
    records = []
    kept_parts = []

    for pid, p_df in df_main.groupby('participant_ID'):
        p_df = assign_blocks(p_df)
        for blk, b_df in p_df.groupby('Block'):
            passed, c, w, m = validate_block(b_df)
            records.append({
                'participant_ID': pid,
                'Block': blk,
                'correct_val': c, 'wrong_val': w, 'missed_val': m,
                'passed': passed
            })
            if passed:
                kept_parts.append(b_df)
            else:
                print(f"  EXCLUDED  participant={pid}  block={blk}  "
                      f"(correct={c}, wrong={w}, missed={m})")

    df_valid      = pd.concat(kept_parts, ignore_index=True) if kept_parts else pd.DataFrame()
    exclusion_log = pd.DataFrame(records)
    return df_valid, exclusion_log


# SECTION 4 – RECOGNITION EVENTS (IR)

def extract_recognition(df_valid):
    """
    For every participant, identify:
      - Each target sentence's SECOND (test) showing
      - Whether the participant pressed Spacebar (IR = 1) or not (IR = 0)
      - The RT of that press

    Returns one row per (participant × sentence test trial).
    """
    # All second showings of targets (non-validation)
    second_shown = df_valid[
        (df_valid['isTarget']     == True) &
        (df_valid['isRepeat']     == True) &
        (df_valid['Event']        == 'Sentence shown') &
        (df_valid['isValidation'] != True)
    ][['participant_ID','Stimulus','Condition','SentenceID','Voice','Block']].drop_duplicates(
        subset=['participant_ID','Stimulus']
    ).copy()

    # IR presses on those trials
    ir_pressed = df_valid[
        (df_valid['isTarget']     == True) &
        (df_valid['isRepeat']     == True) &
        (df_valid['Event']        == 'IR pressed') &
        (df_valid['isValidation'] != True)
    ][['participant_ID','Stimulus','Accuracy IR','Reaction_time_IR']].copy()

    ir_pressed['Accuracy IR']      = pd.to_numeric(ir_pressed['Accuracy IR'],      errors='coerce')
    ir_pressed['Reaction_time_IR'] = pd.to_numeric(ir_pressed['Reaction_time_IR'], errors='coerce')

    # Merge: missing IR press = 0 (missed)
    rec = second_shown.merge(ir_pressed, on=['participant_ID','Stimulus'], how='left')
    rec['Hit'] = rec['Accuracy IR'].fillna(0).astype(int)
    rec.rename(columns={'Reaction_time_IR': 'RT_ms'}, inplace=True)
    rec.drop(columns=['Accuracy IR'], inplace=True)

    return rec


# SECTION 5 – FALSE ALARM RATE (per participant)

def compute_fa_rates(df_valid):
    """
    False alarm = pressing Spacebar (IR) on a filler sentence that has
    NOT been seen before (first showing of a FILLER, non-validation).
    fa_rate = n_FA_presses / n_filler_first_showings
    """
    rows = []
    for pid, p_df in df_valid.groupby('participant_ID'):

        # Filler first showings (not validation, not repeat)
        filler_shown = p_df[
            (p_df['Condition']    == 'FILLER') &
            (p_df['isValidation'] != True) &
            (p_df['isRepeat']     != True) &
            (p_df['Event']        == 'Sentence shown')
        ]
        n_filler = len(filler_shown)

        fa_presses = p_df[
            (p_df['Condition']    == 'FILLER') &
            (p_df['isValidation'] != True) &
            (p_df['isRepeat']     != True) &
            (p_df['Event']        == 'IR pressed')      # exact match only
        ]
        n_fa    = len(fa_presses)
        fa_rate = n_fa / n_filler if n_filler > 0 else 0.0

        rows.append({'participant_ID': pid, 'n_filler': n_filler,
                     'n_fa': n_fa, 'fa_rate': fa_rate})

    return pd.DataFrame(rows)


# SECTION 6 – CORRECTED MEMORABILITY (per participant × condition × voice)

def compute_corrected_memorability(rec_df, fa_df):
    """
    For each participant * condition * voice:
      hit_rate   = hits / total_trials
      corr_mem   = hit_rate - participant's FA rate

    Returns a tidy DataFrame with one row per participant * condition * voice.
    """
    rec_targets = rec_df[rec_df['Condition'].isin(CONDITIONS)].copy()

    # Hit rate per participant × condition × voice
    agg = rec_targets.groupby(['participant_ID','Condition','Voice'])['Hit'].agg(
        hits='sum', total='count', hit_rate='mean'
    ).reset_index()

    # Merge FA rate
    agg = agg.merge(fa_df[['participant_ID','fa_rate']], on='participant_ID', how='left')
    agg['corr_mem'] = agg['hit_rate'] - agg['fa_rate']

    return agg


# SECTION 7 – SENTENCE-LEVEL MEMORABILITY SCORES

def compute_sentence_scores(rec_df, fa_df):
    """
    Each sentence gets a memorability score = proportion of participants who
    recognised it (hit rate across participants who saw it as a target),
    corrected by averaging each participant's FA rate.

    This is the unit of analysis for the Kruskal-Wallis test:
    one score per sentence, grouped by condition.
    """
    rec_targets = rec_df[rec_df['Condition'].isin(CONDITIONS)].copy()

    # Merge FA rate onto each trial
    rec_targets = rec_targets.merge(fa_df[['participant_ID','fa_rate']],
                                    on='participant_ID', how='left')

    # Per-sentence: mean hit, mean FA, corrected score
    sent = rec_targets.groupby(['Stimulus','Condition','Voice']).agg(
        n_participants = ('Hit', 'count'),
        mean_hit_rate  = ('Hit', 'mean'),
        mean_fa_rate   = ('fa_rate', 'mean'),
    ).reset_index()
    sent['corr_mem_score'] = sent['mean_hit_rate'] - sent['mean_fa_rate']

    return sent


# SECTION 8 – KRUSKAL-WALLIS + BONFERRONI POST-HOC

def kruskal_wallis_analysis(sent_scores):
    """
    Kruskal-Wallis test on sentence-level corrected memorability scores
    across the four conditions.

    Post-hoc: Mann-Whitney U for all 6 pairs, Bonferroni-corrected.

    Returns:
      kw_result  : dict with H statistic, p-value, df
      posthoc_df : DataFrame of pairwise comparisons
    """
    groups = {c: sent_scores[sent_scores['Condition']==c]['corr_mem_score'].values
              for c in CONDITIONS}

    valid = {c: g for c, g in groups.items() if len(g) > 0}
    if len(valid) < 2:
        print("  Not enough groups for Kruskal-Wallis.")
        return None, pd.DataFrame()

    H, p = stats.kruskal(*valid.values())
    kw_result = {'H': H, 'p': p, 'df': len(valid)-1,
                 'n_sentences_per_cond': {c: len(g) for c,g in valid.items()}}
    print(f"\nKruskal-Wallis: H({len(valid)-1}) = {H:.4f}, p = {p:.4f}")

    # Post-hoc pairwise Mann-Whitney U with Bonferroni correction
    pairs = list(combinations(CONDITIONS, 2))
    n_tests = len(pairs)
    rows = []
    for c1, c2 in pairs:
        g1, g2 = groups.get(c1, np.array([])), groups.get(c2, np.array([]))
        if len(g1) > 0 and len(g2) > 0:
            U, p_raw = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            rows.append({
                'Condition_1': c1, 'Condition_2': c2,
                'n1': len(g1), 'n2': len(g2),
                'U': U, 'p_raw': p_raw,
                'p_bonferroni': min(p_raw * n_tests, 1.0),
                'significant': (p_raw * n_tests) < 0.05
            })
    posthoc_df = pd.DataFrame(rows)
    return kw_result, posthoc_df


# SECTION 9 – WR (VOICE DISCRIMINATION) ANALYSIS

def extract_wr(df_valid):
    """
    Extract Word Recognition (WR) responses:
    Among recognised targets, did the participant correctly identify
    whether the voice was the same (A) or changed (D)?

    Returns one row per recognised trial with WR response.
    """
    wr = df_valid[
        (df_valid['isTarget']     == True) &
        (df_valid['isRepeat']     == True) &
        (df_valid['Event']        == 'WR pressed') &
        (df_valid['isValidation'] != True)
    ][['participant_ID','Stimulus','Condition','Voice',
       'Button','Accuracy WR','Reaction_time_WR']].copy()

    wr['Accuracy WR']      = pd.to_numeric(wr['Accuracy WR'],      errors='coerce')
    wr['Reaction_time_WR'] = pd.to_numeric(wr['Reaction_time_WR'], errors='coerce')
    return wr


# SECTION 10 – DESCRIPTIVE STATS HELPERS

def describe_by_condition(corr_mem_df):
    """
    Aggregate corrected memorability across participants, per condition.
    Each participant contributes one score per condition (averaged over voices).
    This gives N=114 data points per condition.
    """
    # Average over Active/Passive per participant first, then summarise across participants
    per_p = corr_mem_df.groupby(['participant_ID','Condition'])['corr_mem'].mean().reset_index()
    desc = per_p.groupby('Condition')['corr_mem'].agg(
        N='count', Mean='mean', SD='std', Median='median',
        Min='min', Max='max',
        SE=lambda x: x.std() / np.sqrt(len(x))
    ).reindex(CONDITIONS).reset_index()
    return desc


def describe_by_condition_voice(corr_mem_df):
    """
    Aggregate corrected memorability per condition × voice across participants.
    """
    desc = corr_mem_df.groupby(['Condition','Voice'])['corr_mem'].agg(
        N='count', Mean='mean', SD='std', Median='median',
        SE=lambda x: x.std() / np.sqrt(len(x))
    ).reset_index()
    return desc


def fig_participant_overview(exclusion_log, fa_df, rec_df):
    """
    Figure 1: Sample & data quality overview.
      (a) Participants excluded per block (bar chart)
      (b) Distribution of FA rates across participants (histogram)
      (c) Distribution of overall hit rates across participants (histogram)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Figure 1: Sample Overview and Data Quality', fontweight='bold')

    # (a) Block exclusion counts
    excl = exclusion_log.groupby('Block')['passed'].apply(lambda x: (~x).sum()).reset_index()
    excl.columns = ['Block', 'n_excluded']
    axes[0].bar(excl['Block'], excl['n_excluded'], color=COLORS['accent'], alpha=0.8, width=0.5)
    axes[0].set_title('(a) Blocks Excluded per Block Number')
    axes[0].set_xlabel('Block')
    axes[0].set_ylabel('N Participants Excluded')
    axes[0].set_xticks([1, 2, 3])

    # (b) FA rate distribution
    axes[1].hist(fa_df['fa_rate'], bins=20, color=COLORS['HL'], edgecolor='white', alpha=0.85)
    axes[1].axvline(fa_df['fa_rate'].mean(), color='black', linestyle='--',
                    linewidth=1.5, label=f"Mean = {fa_df['fa_rate'].mean():.3f}")
    axes[1].set_title('(b) Distribution of False Alarm Rates')
    axes[1].set_xlabel('False Alarm Rate')
    axes[1].set_ylabel('N Participants')
    axes[1].legend(fontsize=9)

    # (c) Overall hit rate distribution
    overall_hr = rec_df[rec_df['Condition'].isin(CONDITIONS)].groupby(
        'participant_ID')['Hit'].mean().reset_index()
    axes[2].hist(overall_hr['Hit'], bins=20, color=COLORS['LH'], edgecolor='white', alpha=0.85)
    axes[2].axvline(overall_hr['Hit'].mean(), color='black', linestyle='--',
                    linewidth=1.5, label=f"Mean = {overall_hr['Hit'].mean():.3f}")
    axes[2].set_title('(c) Distribution of Overall Hit Rates')
    axes[2].set_xlabel('Hit Rate (proportion)')
    axes[2].set_ylabel('N Participants')
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    savefig('fig1_sample_overview.png')


def fig_corrected_memorability_by_condition(desc_cond):
    """
    Figure 2: Mean corrected memorability ± SE per condition.
    Each data point = one participant's average corrected memorability for that condition.
    Error bars = ±1 SE across participants.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(CONDITIONS))
    colors = [COLORS[c] for c in CONDITIONS]

    bars = ax.bar(x, desc_cond['Mean'],
                  color=colors, width=0.55, edgecolor='white', linewidth=0.8, alpha=0.9)
    ax.errorbar(x, desc_cond['Mean'], yerr=desc_cond['SE'],
                fmt='none', color='black', capsize=5, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    ax.set_xlabel('Condition (Subject Memorability × Object Memorability)')
    ax.set_ylabel('Corrected Memorability (Hit Rate − FA Rate)')
    ax.set_title('Figure 2: Mean Corrected Memorability by Condition\n'
                 f'(N = {int(desc_cond["N"].mean())} participants, error bars = ±1 SE)')

    for bar, val in zip(bars, desc_cond['Mean']):
        ypos = bar.get_height() + desc_cond['SE'].iloc[list(bars).index(bar)] + 0.01
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.3f}',
                ha='center', fontsize=9, fontweight='bold')

    patch_legend = [mpatches.Patch(color=COLORS[c], label=c) for c in CONDITIONS]
    ax.legend(handles=patch_legend, title='Condition', fontsize=9)
    plt.tight_layout()
    savefig('fig2_corrected_memorability_by_condition.png')


def fig_condition_voice(desc_cv):
    """
    Figure 3: Corrected memorability by condition * voice (grouped bar).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CONDITIONS))
    w = 0.35

    for i, voice in enumerate(['Active', 'Passive']):
        sub = desc_cv[desc_cv['Voice'] == voice].set_index('Condition').reindex(CONDITIONS)
        means = sub['Mean'].fillna(0).values
        ses   = sub['SE'].fillna(0).values
        bars  = ax.bar(x + (i - 0.5) * w, means, w,
                       label=voice, color=COLORS[voice], alpha=0.85, edgecolor='white')
        ax.errorbar(x + (i - 0.5) * w, means, yerr=ses,
                    fmt='none', color='black', capsize=4, linewidth=1.2)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    ax.set_xlabel('Condition')
    ax.set_ylabel('Corrected Memorability (mean ± SE)')
    ax.set_title('Figure 3: Corrected Memorability by Condition * Voice')
    ax.legend(title='Voice')
    plt.tight_layout()
    savefig('fig3_condition_by_voice.png')


def fig_sentence_score_distribution(sent_scores):
    """
    Figure 4: Distribution of sentence-level corrected memorability scores
    per condition (violin + strip plot).
    These are the scores used in the Kruskal-Wallis test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figure 4: Sentence-Level Memorability Score Distributions', fontweight='bold')

    # 4a: Violin by condition
    data_by_cond = [sent_scores[sent_scores['Condition'] == c]['corr_mem_score'].values
                    for c in CONDITIONS]
    vp = axes[0].violinplot(data_by_cond, positions=range(len(CONDITIONS)),
                            showmedians=True, showextrema=True)
    for i, (body, c) in enumerate(zip(vp['bodies'], CONDITIONS)):
        body.set_facecolor(COLORS[c])
        body.set_alpha(0.7)
    for comp in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        vp[comp].set_color('black')
        vp[comp].set_linewidth(1.5)
    axes[0].set_xticks(range(len(CONDITIONS)))
    axes[0].set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    axes[0].axhline(0, color='grey', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Condition')
    axes[0].set_ylabel('Corrected Memorability Score (per sentence)')
    axes[0].set_title('(a) By Condition')

    # 4b: Violin by voice within condition (Active vs Passive)
    for j, voice in enumerate(['Active', 'Passive']):
        sub = sent_scores[sent_scores['Voice'] == voice]
        vals = [sub[sub['Condition'] == c]['corr_mem_score'].values for c in CONDITIONS]
        offset = -0.2 + j * 0.4
        bp = axes[1].boxplot(vals,
                             positions=[k + offset for k in range(len(CONDITIONS))],
                             widths=0.3, patch_artist=True,
                             medianprops={'color': 'black', 'linewidth': 2},
                             boxprops={'facecolor': COLORS[voice], 'alpha': 0.7})
    axes[1].set_xticks(range(len(CONDITIONS)))
    axes[1].set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    axes[1].axhline(0, color='grey', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Condition')
    axes[1].set_ylabel('Corrected Memorability Score (per sentence)')
    axes[1].set_title('(b) By Condition × Voice')
    legend_patches = [mpatches.Patch(color=COLORS[v], label=v, alpha=0.7)
                      for v in ['Active', 'Passive']]
    axes[1].legend(handles=legend_patches)

    plt.tight_layout()
    savefig('fig4_sentence_score_distributions.png')


def fig_rt_distribution(rec_df):
    """
    Figure 5: Reaction time distributions for correct hits.
    (a) By condition  (b) By voice
    Based on all participants' trial-level RTs.
    """
    rt_df = rec_df[(rec_df['Condition'].isin(CONDITIONS)) & (rec_df['Hit'] == 1)].copy()
    rt_df['RT_ms'] = pd.to_numeric(rt_df['RT_ms'], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figure 5: Reaction Times for Correct Recognitions (All Participants)',
                 fontweight='bold')

    # 5a: Boxplot by condition
    data_cond = [rt_df[rt_df['Condition'] == c]['RT_ms'].dropna().values for c in CONDITIONS]
    bp1 = axes[0].boxplot(data_cond, patch_artist=True,
                          medianprops={'color': 'black', 'linewidth': 2},
                          flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3})
    for patch, c in zip(bp1['boxes'], CONDITIONS):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.75)
    axes[0].set_xticks(range(1, len(CONDITIONS) + 1))
    axes[0].set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    axes[0].set_xlabel('Condition')
    axes[0].set_ylabel('Reaction Time (ms)')
    axes[0].set_title('(a) RT by Condition')
    for i, vals in enumerate(data_cond):
        if len(vals) > 0:
            axes[0].text(i + 1, np.percentile(vals, 75) + 50,
                         f'M={np.mean(vals):.0f}', ha='center', fontsize=8, color='grey')

    # 5b: Boxplot by voice
    data_voice = [rt_df[rt_df['Voice'] == v]['RT_ms'].dropna().values
                  for v in ['Active', 'Passive']]
    bp2 = axes[1].boxplot(data_voice, patch_artist=True,
                          medianprops={'color': 'black', 'linewidth': 2},
                          flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3})
    for patch, v in zip(bp2['boxes'], ['Active', 'Passive']):
        patch.set_facecolor(COLORS[v])
        patch.set_alpha(0.75)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['Active', 'Passive'])
    axes[1].set_xlabel('Voice')
    axes[1].set_ylabel('Reaction Time (ms)')
    axes[1].set_title('(b) RT by Voice')

    plt.tight_layout()
    savefig('fig5_reaction_times.png')


def fig_wr_accuracy(wr_df):
    """
    Figure 6: Word Recognition (voice discrimination) accuracy.
    (a) Mean WR accuracy per condition across participants
    (b) Mean WR accuracy by voice at test
    """
    wr_targets = wr_df[wr_df['Condition'].isin(CONDITIONS)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Figure 6: Voice Discrimination (WR) Accuracy', fontweight='bold')

    # (a) By condition
    wr_cond = wr_targets.groupby(['participant_ID', 'Condition'])['Accuracy WR'].mean().reset_index()
    wr_cond_desc = wr_cond.groupby('Condition')['Accuracy WR'].agg(
        Mean='mean', SE=lambda x: x.std() / np.sqrt(len(x))
    ).reindex(CONDITIONS).reset_index()

    bars_a = axes[0].bar(range(len(CONDITIONS)), wr_cond_desc['Mean'],
                         color=[COLORS[c] for c in CONDITIONS],
                         edgecolor='white', width=0.55, alpha=0.9)
    axes[0].errorbar(range(len(CONDITIONS)), wr_cond_desc['Mean'],
                     yerr=wr_cond_desc['SE'],
                     fmt='none', color='black', capsize=5, linewidth=1.5)
    axes[0].axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Chance (0.50)')
    axes[0].set_xticks(range(len(CONDITIONS)))
    axes[0].set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    axes[0].set_ylim(0, 1.1)
    axes[0].set_xlabel('Condition')
    axes[0].set_ylabel('Mean WR Accuracy (± SE)')
    axes[0].set_title('(a) Voice Discrimination by Condition')
    axes[0].legend(fontsize=9)

    # (b) By voice at test
    wr_voice = wr_targets.groupby(['participant_ID', 'Voice'])['Accuracy WR'].mean().reset_index()
    wr_voice_desc = wr_voice.groupby('Voice')['Accuracy WR'].agg(
        Mean='mean', SE=lambda x: x.std() / np.sqrt(len(x))
    ).reset_index()

    bars_b = axes[1].bar(range(len(wr_voice_desc)), wr_voice_desc['Mean'],
                         color=[COLORS.get(v, 'grey') for v in wr_voice_desc['Voice']],
                         edgecolor='white', width=0.4, alpha=0.9)
    axes[1].errorbar(range(len(wr_voice_desc)), wr_voice_desc['Mean'],
                     yerr=wr_voice_desc['SE'],
                     fmt='none', color='black', capsize=5, linewidth=1.5)
    axes[1].axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Chance (0.50)')
    axes[1].set_xticks(range(len(wr_voice_desc)))
    axes[1].set_xticklabels(wr_voice_desc['Voice'])
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlabel('Voice at Test')
    axes[1].set_ylabel('Mean WR Accuracy (± SE)')
    axes[1].set_title('(b) Voice Discrimination by Voice')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    savefig('fig6_wr_accuracy.png')


def fig_kruskal_wallis_summary(kw_result, posthoc_df, sent_scores):
    """
    Figure 7: Inference summary.
    (a) Sentence-level corrected memorability scores by condition (boxplot)
        with KW result annotated
    (b) Heatmap of Bonferroni-corrected p-values for pairwise comparisons
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figure 7: Kruskal-Wallis Analysis Summary', fontweight='bold')

    # (a) Sentence scores boxplot with KW annotation
    data = [sent_scores[sent_scores['Condition'] == c]['corr_mem_score'].values
            for c in CONDITIONS]
    bp = axes[0].boxplot(data, patch_artist=True,
                         medianprops={'color': 'black', 'linewidth': 2})
    for patch, c in zip(bp['boxes'], CONDITIONS):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.75)
    axes[0].set_xticks(range(1, len(CONDITIONS) + 1))
    axes[0].set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    axes[0].axhline(0, color='grey', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Condition')
    axes[0].set_ylabel('Corrected Memorability (per sentence)')
    axes[0].set_title('(a) Sentence Scores by Condition')
    if kw_result:
        sig = '***' if kw_result['p'] < 0.001 else ('**' if kw_result['p'] < 0.01
              else ('*' if kw_result['p'] < 0.05 else 'n.s.'))
        axes[0].text(0.97, 0.97,
                     f"KW H({kw_result['df']}) = {kw_result['H']:.2f}\np = {kw_result['p']:.4f} {sig}",
                     transform=axes[0].transAxes, ha='right', va='top',
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (b) Bonferroni p-value heatmap
    if not posthoc_df.empty:
        p_matrix = pd.DataFrame(np.ones((4, 4)), index=CONDITIONS, columns=CONDITIONS)
        for _, row in posthoc_df.iterrows():
            p_matrix.loc[row['Condition_1'], row['Condition_2']] = row['p_bonferroni']
            p_matrix.loc[row['Condition_2'], row['Condition_1']] = row['p_bonferroni']
        mask = np.triu(np.ones_like(p_matrix, dtype=bool))
        sns.heatmap(p_matrix, mask=mask, annot=True, fmt='.3f',
                    cmap='RdYlGn_r', vmin=0, vmax=1, ax=axes[1],
                    linewidths=0.5, cbar_kws={'label': 'Bonferroni-corrected p'})
        axes[1].set_title('(b) Pairwise p-values (Bonferroni-corrected)\n'
                          '(Green = significant at α=0.05)')

    plt.tight_layout()
    savefig('fig7_kruskal_wallis.png')


def save_outputs(corr_mem_df, desc_cond, desc_cv, sent_scores,
                 fa_df, exclusion_log, kw_result, posthoc_df, wr_df):

    corr_mem_df.to_csv(os.path.join(OUTPUT_FOLDER, 'corrected_memorability_per_participant.csv'), index=False)
    desc_cond.to_csv(os.path.join(OUTPUT_FOLDER, 'descriptive_stats_by_condition.csv'), index=False)
    desc_cv.to_csv(os.path.join(OUTPUT_FOLDER, 'descriptive_stats_by_condition_voice.csv'), index=False)
    sent_scores.to_csv(os.path.join(OUTPUT_FOLDER, 'sentence_level_scores.csv'), index=False)
    fa_df.to_csv(os.path.join(OUTPUT_FOLDER, 'false_alarm_rates.csv'), index=False)
    exclusion_log.to_csv(os.path.join(OUTPUT_FOLDER, 'validation_exclusion_log.csv'), index=False)
    posthoc_df.to_csv(os.path.join(OUTPUT_FOLDER, 'posthoc_pairwise_tests.csv'), index=False)

    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS – Corrected Memorability by Condition")
    print("="*60)
    print(desc_cond.to_string(index=False))

    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS – By Condition × Voice")
    print("="*60)
    print(desc_cv.to_string(index=False))

    if kw_result:
        print("\n" + "="*60)
        print("KRUSKAL-WALLIS TEST RESULT")
        print("="*60)
        print(f"  H({kw_result['df']}) = {kw_result['H']:.4f},  p = {kw_result['p']:.4f}")
        print(f"  N sentences per condition: {kw_result['n_sentences_per_cond']}")

    print("\n" + "="*60)
    print("POST-HOC PAIRWISE COMPARISONS (Bonferroni-corrected)")
    print("="*60)
    print(posthoc_df.to_string(index=False))

    print(f"\nAll outputs saved to: {OUTPUT_FOLDER}")


def main():
    print("\n[1] Loading log files...")
    df_all = load_all_logs(DATA_FOLDER)

    print("\n[2] Separating practice from main experiment...")
    df_main = df_all[~df_all['is_practice']].copy()
    print(f"    Main experiment rows: {len(df_main)}")

    print("\n[3] Applying per-block validation exclusion...")
    df_valid, exclusion_log = apply_validation_exclusion(df_main)
    n_excl = (~exclusion_log['passed']).sum()
    print(f"    Blocks excluded: {n_excl} / {len(exclusion_log)} total blocks "
          f"({100*n_excl/len(exclusion_log):.1f}%)")

    print("\n[4] Extracting recognition (IR) trials...")
    rec_df = extract_recognition(df_valid)
    print(f"    Total recognition trials: {len(rec_df)}")
    print(f"    Overall hit rate: {rec_df['Hit'].mean():.3f}")

    print("\n[5] Computing false alarm rates per participant...")
    fa_df = compute_fa_rates(df_valid)
    print(f"    Mean FA rate: {fa_df['fa_rate'].mean():.4f}  "
          f"(SD={fa_df['fa_rate'].std():.4f})")

    print("\n[6] Computing corrected memorability scores...")
    corr_mem_df = compute_corrected_memorability(rec_df, fa_df)

    print("\n[7] Computing descriptive statistics...")
    desc_cond = describe_by_condition(corr_mem_df)
    desc_cv   = describe_by_condition_voice(corr_mem_df)

    print("\n[8] Computing sentence-level memorability scores...")
    sent_scores = compute_sentence_scores(rec_df, fa_df)
    print(f"    Total sentences scored: {len(sent_scores)}")

    print("\n[9] Extracting word recognition (WR) data...")
    wr_df = extract_wr(df_valid)
    print(f"    WR trials: {len(wr_df)}")

    print("\n[10] Running Kruskal-Wallis test and post-hoc comparisons...")
    kw_result, posthoc_df = kruskal_wallis_analysis(sent_scores)

    print("\n[11] Generating figures...")
    fig_participant_overview(exclusion_log, fa_df, rec_df)
    fig_corrected_memorability_by_condition(desc_cond)
    fig_condition_voice(desc_cv)
    fig_sentence_score_distribution(sent_scores)
    fig_rt_distribution(rec_df)
    fig_wr_accuracy(wr_df)
    fig_kruskal_wallis_summary(kw_result, posthoc_df, sent_scores)

    print("\n[12] Saving CSV outputs...")
    save_outputs(corr_mem_df, desc_cond, desc_cv, sent_scores,
                 fa_df, exclusion_log, kw_result, posthoc_df, wr_df)


if __name__ == '__main__':
    main()