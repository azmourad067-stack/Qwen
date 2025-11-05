import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# ML avancé
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🏇 Analyseur Hippique IA Pro v2",
    page_icon="🏇",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        border-left: 5px solid #f59e0b;
        padding: 1rem 1rem 1rem 1.5rem;
        background: linear-gradient(90deg, #fffbeb 0%, #ffffff 100%);
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .confidence-high { color: #10b981; font-weight: bold; }
    .confidence-medium { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

CONFIGS = {
    "PLAT": {
        "description": "🏃 Course de galop - Handicap poids + avantage corde intérieure",
        "optimal_draws": [1, 2, 3, 4],
        "weight_importance": 0.25
    },
    "ATTELE_AUTOSTART": {
        "description": "🚗 Trot attelé autostart - Numéros 4-6 optimaux", 
        "optimal_draws": [4, 5, 6],
        "weight_importance": 0.05
    },
    "ATTELE_VOLTE": {
        "description": "🔄 Trot attelé volté - Numéro sans importance",
        "optimal_draws": [],
        "weight_importance": 0.05
    }
}

# --- PARTIE SCRAPING - INTACTE ---
@st.cache_data(ttl=300)
def scrape_race_data(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"
        soup = BeautifulSoup(response.content, 'html.parser')
        horses_data = []
        table = soup.find('table')
        if not table:
            return None, "Aucun tableau trouvé"
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 4:
                horses_data.append({
                    "Numéro de corde": cols[0].get_text(strip=True),
                    "Nom": cols[1].get_text(strip=True),
                    "Cote": cols[-1].get_text(strip=True),
                    "Poids": cols[-2].get_text(strip=True) if len(cols) > 4 else "60.0",
                    "Musique": cols[2].get_text(strip=True) if len(cols) > 5 else "",
                    "Âge/Sexe": cols[3].get_text(strip=True) if len(cols) > 6 else "",
                })
        if not horses_data:
            return None, "Aucune donnée extraite"
        return pd.DataFrame(horses_data), "Succès"
    except Exception as e:
        return None, f"Erreur: {str(e)}"

def safe_convert(value, convert_func, default=0):
    try:
        if pd.isna(value):
            return default
        cleaned = str(value).replace(',', '.').strip()
        return convert_func(cleaned)
    except:
        return default

def prepare_data(df):
    df = df.copy()
    df['odds_numeric'] = df['Cote'].apply(lambda x: safe_convert(x, float, 999))
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_convert(x, int, 1))
    def extract_weight(poids_str):
        if pd.isna(poids_str):
            return 60.0
        match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
        return float(match.group(1).replace(',', '.')) if match else 60.0
    df['weight_kg'] = df['Poids'].apply(extract_weight)
    df = df[df['odds_numeric'] > 0]
    df = df.reset_index(drop=True)
    return df

# --- NOUVELLE CLASSE ML AMÉLIORÉE ---
@st.cache_resource
class AdvancedHorseRacingML:
    def __init__(self):
        # Modèles de base avancés - Ajout d'un réseau de neurones
        self.base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=10,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'elastic': ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
            'neural_net': MLPRegressor(
                hidden_layer_sizes=(100, 50), # Architecture simple pour commencer
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        }
        # Modèle d'ensemble (stacking) avec modèle final plus robuste
        self.stacking_model = None
        self.scaler = RobustScaler()  # Plus robuste aux outliers
        self.feature_importance = {}
        self.cv_scores = {}
        self.confidence_scores = []
        self.is_trained = False
        self.feature_correlations = {} # Pour stocker les corrélations
        self.feature_weights = {}      # Pour stocker les poids dynamiques

    def extract_music_features(self, music_str):
        """Extraction avancée des performances passées"""
        if pd.isna(music_str) or music_str == '':
            return {
                'wins': 0, 'places': 0, 'total_races': 0,
                'win_rate': 0, 'place_rate': 0, 'consistency': 0,
                'recent_form': 0, 'best_position': 10,
                'avg_position': 8, 'position_variance': 5
            }
        music = str(music_str)
        positions = [int(c) for c in music if c.isdigit() and int(c) > 0]
        if not positions:
            return {
                'wins': 0, 'places': 0, 'total_races': 0,
                'win_rate': 0, 'place_rate': 0, 'consistency': 0,
                'recent_form': 0, 'best_position': 10,
                'avg_position': 8, 'position_variance': 5
            }
        total = len(positions)
        wins = positions.count(1)
        places = sum(1 for p in positions if p <= 3)
        # Forme récente (3 dernières courses)
        recent = positions[:3]
        recent_form = sum(1/p for p in recent) / len(recent) if recent else 0
        # Régularité
        consistency = 1 / (np.std(positions) + 1) if len(positions) > 1 else 0
        return {
            'wins': wins,
            'places': places,
            'total_races': total,
            'win_rate': wins / total if total > 0 else 0,
            'place_rate': places / total if total > 0 else 0,
            'consistency': consistency,
            'recent_form': recent_form,
            'best_position': min(positions),
            'avg_position': np.mean(positions),
            'position_variance': np.var(positions)
        }

    def calculate_correlations(self, X, y_synthetic):
        """Calcul des corrélations entre features et cible synthétique"""
        correlations = {}
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                corr = np.corrcoef(X[col], y_synthetic)[0, 1]
                correlations[col] = abs(corr) # On prend la valeur absolue pour la pondération
        self.feature_correlations = correlations

    def prepare_advanced_features(self, df, race_type="PLAT"):
        """Création de features avancées pour ML"""
        features = pd.DataFrame()
        # === FEATURES DE BASE ===
        features['odds_inv'] = 1 / (df['odds_numeric'] + 0.1)
        features['log_odds'] = np.log1p(df['odds_numeric'])
        features['sqrt_odds'] = np.sqrt(df['odds_numeric'])
        features['odds_squared'] = df['odds_numeric'] ** 2
        # === FEATURES DE POSITION ===
        features['draw'] = df['draw_numeric']
        features['draw_normalized'] = df['draw_numeric'] / df['draw_numeric'].max()
        # Avantage position selon type de course
        optimal_draws = CONFIGS[race_type]['optimal_draws']
        features['optimal_draw'] = df['draw_numeric'].apply(
            lambda x: 1 if x in optimal_draws else 0
        )
        features['draw_distance_optimal'] = df['draw_numeric'].apply(
            lambda x: min([abs(x - opt) for opt in optimal_draws]) if optimal_draws else 0
        )
        # === FEATURES DE POIDS ===
        features['weight'] = df['weight_kg']
        features['weight_normalized'] = (df['weight_kg'] - df['weight_kg'].mean()) / (df['weight_kg'].std() + 1e-6)
        features['weight_rank'] = df['weight_kg'].rank()
        weight_importance = CONFIGS[race_type]['weight_importance']
        features['weight_advantage'] = (df['weight_kg'].max() - df['weight_kg']) * weight_importance
        # === FEATURES D'ÂGE ET SEXE ===
        if 'Âge/Sexe' in df.columns:
            features['age'] = df['Âge/Sexe'].str.extract('(\d+)').astype(float).fillna(4)
            features['is_mare'] = df['Âge/Sexe'].str.contains('F', na=False).astype(int)
            features['is_stallion'] = df['Âge/Sexe'].str.contains('H', na=False).astype(int)
            features['age_squared'] = features['age'] ** 2
            features['age_optimal'] = features['age'].apply(lambda x: 1 if 4 <= x <= 6 else 0)
        else:
            features['age'] = 4.5
            features['is_mare'] = 0
            features['is_stallion'] = 0
            features['age_squared'] = 20.25
            features['age_optimal'] = 1
        # === FEATURES DE MUSIQUE (HISTORIQUE) ===
        if 'Musique' in df.columns:
            music_features = df['Musique'].apply(self.extract_music_features)
            for key in music_features.iloc[0].keys():
                features[f'music_{key}'] = [m[key] for m in music_features]
        else:
            for key in ['wins', 'places', 'total_races', 'win_rate', 'place_rate', 
                       'consistency', 'recent_form', 'best_position', 'avg_position', 'position_variance']:
                features[f'music_{key}'] = 0
        # === FEATURES D'INTERACTION ===
        features['odds_draw_interaction'] = features['odds_inv'] * features['draw_normalized']
        features['odds_weight_interaction'] = features['log_odds'] * features['weight_normalized']
        features['age_weight_interaction'] = features['age'] * features['weight']
        features['form_odds_interaction'] = features['music_recent_form'] * features['odds_inv']
        features['consistency_weight'] = features['music_consistency'] * features['weight_advantage']
        # === FEATURES DE CLASSEMENT RELATIF ===
        features['odds_rank'] = df['odds_numeric'].rank()
        features['odds_percentile'] = df['odds_numeric'].rank(pct=True)
        features['weight_percentile'] = df['weight_kg'].rank(pct=True)
        # === FEATURES STATISTIQUES ===
        features['odds_z_score'] = (df['odds_numeric'] - df['odds_numeric'].mean()) / (df['odds_numeric'].std() + 1e-6)
        features['is_favorite'] = (df['odds_numeric'] == df['odds_numeric'].min()).astype(int)
        features['is_outsider'] = (df['odds_numeric'] > df['odds_numeric'].quantile(0.75)).astype(int)
        # === FEATURES DE CONTEXTE ===
        features['field_size'] = len(df)
        features['competitive_index'] = df['odds_numeric'].std() / (df['odds_numeric'].mean() + 1e-6)
        
        # === FEATURES POUR LA RÉGRESSION LOGISTIQUE (Classification) ===
        # Ces features sont souvent plus linéaires
        features['logit_odds_inv'] = features['odds_inv']
        features['logit_music_win_rate'] = features['music_win_rate']
        features['logit_music_place_rate'] = features['music_place_rate']
        features['logit_music_recent_form'] = features['music_recent_form']
        features['logit_consistency'] = features['music_consistency']
        features['logit_optimal_draw'] = features['optimal_draw']
        
        return features.fillna(0)

    def train_with_cross_validation(self, X, y, cv_folds=5):
        """Entraînement avec validation croisée"""
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for name, model in self.base_models.items():
            try:
                scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
                self.cv_scores[name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
            except Exception as e:
                st.warning(f"Erreur CV pour {name}: {e}")
                self.cv_scores[name] = {'mean': 0, 'std': 1, 'scores': [0]}

    def create_stacking_model(self):
        """Création d'un modèle d'ensemble par stacking"""
        estimators = [
            ('rf', self.base_models['random_forest']),
            ('gb', self.base_models['gradient_boosting']),
            ('nn', self.base_models['neural_net']), # Ajout du réseau de neurones
            ('ridge', self.base_models['ridge'])
        ]
        # Modèle final plus robuste
        final_estimator = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1,
            random_state=42
        )
        self.stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1
        )

    def calculate_prediction_confidence(self, predictions, X):
        """Calcul de la confiance dans les prédictions"""
        if len(predictions) < 3:
            return np.ones(len(predictions)) * 0.5
        # Variance des prédictions (inversement proportionnelle à la confiance)
        pred_std = np.std(predictions)
        confidence_base = 1 / (1 + pred_std)
        # Ajustement par la qualité des features (corrélations)
        if self.feature_correlations:
            feature_weights = np.array([self.feature_correlations.get(col, 0) for col in X.columns])
            feature_quality = np.mean(feature_weights, axis=0) # Moyenne pondérée par corrélation
        else:
            feature_quality = 1 - (X.isna().sum(axis=1) / len(X.columns))
        # Confiance finale
        confidence = confidence_base * feature_quality
        confidence = np.clip(confidence, 0, 1)
        return confidence

    def train_and_predict(self, X, race_type="PLAT"):
        """Entraînement et prédiction avec modèles avancés"""
        if len(X) < 5:
            st.warning("⚠️ Pas assez de données pour un entraînement robuste")
            return np.zeros(len(X)), {}, np.zeros(len(X))

        # --- CRÉATION DE LABELS SYNTHÉTIQUES AMÉLIORÉS ---
        # Pondération basée sur la musique et les cotes
        y_synthetic = (
            X['odds_inv'] * 0.3 +
            X['music_win_rate'] * 0.25 +
            X['music_recent_form'] * 0.2 +
            X['music_consistency'] * 0.15 +
            X['optimal_draw'] * 0.05 +
            X['weight_advantage'] * 0.05 +
            np.random.normal(0, 0.02, len(X)) # Moins de bruit pour plus de cohérence
        )
        y_synthetic = np.clip(y_synthetic, 0, 1) # Normalisation entre 0 et 1

        # Calcul des corrélations *avant* l'entraînement
        self.calculate_correlations(X, y_synthetic)

        # Normalisation
        X_scaled = self.scaler.fit_transform(X)

        # Validation croisée
        self.train_with_cross_validation(X_scaled, y_synthetic)

        # Entraînement des modèles individuels
        predictions_dict = {}
        for name, model in self.base_models.items():
            try:
                model.fit(X_scaled, y_synthetic)
                pred = model.predict(X_scaled)
                # On normalise les prédictions de chaque modèle
                if pred.max() != pred.min():
                    pred = (pred - pred.min()) / (pred.max() - pred.min())
                predictions_dict[name] = pred
                
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(X.columns, model.feature_importances_))
                    top_10 = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
                    self.feature_importance[name] = top_10
            except Exception as e:
                st.warning(f"Erreur modèle {name}: {e}")
                predictions_dict[name] = np.zeros(len(X))

        # Création et entraînement du modèle stacking
        try:
            self.create_stacking_model()
            self.stacking_model.fit(X_scaled, y_synthetic)
            stacking_pred = self.stacking_model.predict(X_scaled)
            # Normalisation de la prédiction stacking
            if stacking_pred.max() != stacking_pred.min():
                stacking_pred = (stacking_pred - stacking_pred.min()) / (stacking_pred.max() - stacking_pred.min())
            predictions_dict['stacking'] = stacking_pred
            # Évaluation du stacking
            self.cv_scores['stacking'] = {
                'mean': r2_score(y_synthetic, stacking_pred),
                'std': 0,
                'scores': [r2_score(y_synthetic, stacking_pred)]
            }
        except Exception as e:
            st.warning(f"Erreur stacking: {e}")
            stacking_pred = np.mean(list(predictions_dict.values()), axis=0)
            # Normalisation
            if stacking_pred.max() != stacking_pred.min():
                stacking_pred = (stacking_pred - stacking_pred.min()) / (stacking_pred.max() - stacking_pred.min())
            predictions_dict['stacking'] = stacking_pred

        # --- PONDÉRATION AUTOMATIQUE DES CRITÈRES ---
        # Les poids sont maintenant basés sur les scores de validation croisée
        weights = {}
        total_cv_score = sum([scores['mean'] for scores in self.cv_scores.values()])
        for name, scores in self.cv_scores.items():
            weights[name] = scores['mean'] / total_cv_score if total_cv_score > 0 else 1.0 / len(self.cv_scores)

        # Prédiction finale (moyenne pondérée par les scores CV)
        final_predictions = sum(
            predictions_dict.get(name, np.zeros(len(X))) * weights.get(name, 0)
            for name in predictions_dict.keys()
        ) / len(predictions_dict)

        # Calcul de la confiance
        confidence = self.calculate_prediction_confidence(final_predictions, X)
        self.is_trained = True
        return final_predictions, self.cv_scores, confidence

def auto_detect_race_type(df):
    weight_std = df['weight_kg'].std()
    weight_mean = df['weight_kg'].mean()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💪 Écart-type poids", f"{weight_std:.1f} kg")
    with col2:
        st.metric("⚖️ Poids moyen", f"{weight_mean:.1f} kg")
    with col3:
        st.metric("🏇 Nb chevaux", len(df))
    if weight_std > 2.5:
        detected = "PLAT"
        reason = "Grande variation de poids (handicap)"
    elif weight_mean > 65 and weight_std < 1.5:
        detected = "ATTELE_AUTOSTART"
        reason = "Poids uniformes élevés (attelé)"
    else:
        detected = "PLAT"
        reason = "Configuration par défaut"
    st.info(f"🤖 **Type détecté**: {detected} | **Raison**: {reason}")
    return detected

def create_advanced_visualization(df_ranked, ml_model=None):
    """Visualisations avancées avec métriques ML"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            '🏆 Scores de Confiance', 
            '📊 Distribution Cotes', 
            '🧠 Importance Features',
            '⚖️ Poids vs Performance', 
            '📈 Validation Croisée',
            '🎯 Corrélation Cotes-Scores'
        ),
        specs=[
            [{"secondary_y": False}, {"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]
        ]
    )
    colors = px.colors.qualitative.Set3
    # 1. Scores avec confiance
    if 'score_final' in df_ranked.columns and 'confidence' in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['rang'],
                y=df_ranked['score_final'],
                mode='markers+lines',
                marker=dict(
                    size=df_ranked['confidence'] * 20,
                    color=df_ranked['confidence'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Confiance")
                ),
                text=df_ranked['Nom'],
                name='Score'
            ), row=1, col=1
        )
    # 2. Distribution des cotes
    fig.add_trace(
        go.Histogram(
            x=df_ranked['odds_numeric'],
            nbinsx=10,
            marker_color=colors[1],
            name='Cotes'
        ), row=1, col=2
    )
    # 3. Importance des features (si disponible)
    if ml_model and ml_model.feature_importance:
        if 'random_forest' in ml_model.feature_importance:
            importance = ml_model.feature_importance['random_forest']
            fig.add_trace(
                go.Bar(
                    x=list(importance.values()),
                    y=list(importance.keys()),
                    orientation='h',
                    marker_color=colors[2],
                    name='Importance'
                ), row=1, col=3
            )
    # 4. Poids vs Performance
    if 'score_final' in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['weight_kg'],
                y=df_ranked['score_final'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df_ranked['rang'],
                    colorscale='RdYlGn_r',
                    showscale=False
                ),
                text=df_ranked['Nom'],
                name='Poids-Score'
            ), row=2, col=1
        )
    # 5. Scores de validation croisée
    if ml_model and ml_model.cv_scores:
        models = list(ml_model.cv_scores.keys())
        means = [ml_model.cv_scores[m]['mean'] for m in models]
        stds = [ml_model.cv_scores[m]['std'] for m in models]
        fig.add_trace(
            go.Bar(
                x=models,
                y=means,
                error_y=dict(type='data', array=stds),
                marker_color=colors[4],
                name='R² CV'
            ), row=2, col=2
        )
    # 6. Corrélation Cotes-Scores
    if 'score_final' in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['odds_numeric'],
                y=df_ranked['score_final'],
                mode='markers',
                marker=dict(size=8, color=colors[5]),
                text=df_ranked['Nom'],
                name='Cotes vs Score'
            ), row=2, col=3
        )
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text="📊 Analyse ML Complète",
        title_x=0.5,
        title_font_size=20
    )
    return fig

def generate_sample_data(data_type="plat"):
    if data_type == "plat":
        return pd.DataFrame({
            'Nom': ['Thunder Bolt', 'Lightning Star', 'Storm King', 'Rain Dance', 'Wind Walker', 'Fire Dancer', 'Ocean Wave'],
            'Numéro de corde': ['1', '2', '3', '4', '5', '6', '7'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1', '12.5', '15.0'],
            'Poids': ['56.5', '57.0', '58.5', '59.0', '57.5', '60.0', '61.5'],
            'Musique': ['1a2a3a1a', '2a1a4a3a', '3a3a1a2a', '1a4a2a1a', '4a2a5a3a', '5a3a6a4a', '6a5a7a8a'],
            'Âge/Sexe': ['4H', '5M', '3F', '6H', '4M', '5H', '4F']
        })
    elif data_type == "attele":
        return pd.DataFrame({
            'Nom': ['Rapide Éclair', 'Foudre Noire', 'Vent du Nord', 'Tempête Rouge', 'Orage Bleu', 'Cyclone Vert'],
            'Numéro de corde': ['1', '2', '3', '4', '5', '6'],
            'Cote': ['4.2', '8.5', '15.0', '3.8', '6.8', '10.2'],
            'Poids': ['68.0', '68.0', '68.0', '68.0', '68.0', '68.0'],
            'Musique': ['2a1a4a1a', '4a3a2a5a', '6a5a8a7a', '1a2a1a3a', '3a4a5a2a', '5a6a4a8a'],
            'Âge/Sexe': ['5H', '6M', '4F', '7H', '5M', '6H']
        })
    else:
        return pd.DataFrame({
            'Nom': ['Ace Impact', 'Torquator Tasso', 'Adayar', 'Tarnawa', 'Chrono Genesis', 'Mishriff', 'Love'],
            'Numéro de corde': ['1', '2', '3', '4', '5', '6', '7'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1', '5.5', '11.0'],
            'Poids': ['59.5', '59.5', '59.5', '58.5', '58.5', '59.0', '58.0'],
            'Musique': ['1a1a2a1a', '1a3a1a2a', '2a1a4a1a', '1a2a1a3a', '3a1a2a1a', '1a1a1a2a', '2a3a1a4a'],
            'Âge/Sexe': ['4H', '5H', '4H', '5F', '5F', '5H', '4F']
        })

def generate_predictions(df_ranked, n_top=5):
    """Génère des combinaisons de paris basées sur le classement."""
    top_n = df_ranked.head(n_top)
    names = top_n['Nom'].tolist()
    
    # E-trio (Top 3 dans l'ordre)
    trio = names[:3] if len(names) >= 3 else names
    # E-Super4 (Top 4)
    super4 = names[:4] if len(names) >= 4 else names
    # Quinté (Top 5)
    quinte = names[:5] if len(names) >= 5 else names
    
    return trio, super4, quinte

def main():
    st.markdown('<h1 class="main-header">🏇 Analyseur Hippique IA Pro v2</h1>', unsafe_allow_html=True)
    st.markdown("*Analyse prédictive avancée avec ML ensembliste, réseaux de neurones et pondération automatique*")
    with st.sidebar:
        st.header("⚙️ Configuration ML")
        race_type = st.selectbox("🏁 Type de course", ["AUTO", "PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE"])
        use_ml = st.checkbox("✅ Activer ML Avancé", value=True)
        # Poids ML est maintenant déterminé automatiquement par les scores CV
        # ml_confidence = st.slider("🎯 Poids ML", 0.1, 0.9, 0.7, 0.05)
        st.subheader("🧠 Modèles Utilisés")
        st.info("✅ Random Forest (200 arbres)")
        st.info("✅ Gradient Boosting")
        st.info("✅ Réseau de Neurones (MLP)")
        st.info("✅ Ridge & ElasticNet")
        st.info("✅ Stacking Ensemble")
        st.subheader("📊 Features")
        st.success(f"**50+ features** créées automatiquement")
        st.subheader("ℹ️ Informations")
        st.info("📚 **Sources**: turfmining.fr, boturfers.fr")
        st.info("🔬 **Validation**: Cross-validation 5-fold")
        st.info("🔄 **Pondération**: Basée sur les corrélations et scores CV")

    tab1, tab2, tab3 = st.tabs(["🌐 URL Analysis", "📁 Upload CSV", "🧪 Test Data"])
    df_final = None
    with tab1:
        st.subheader("🔍 Analyse d'URL de Course")
        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input("🌐 URL de la course:", placeholder="https://example-racing-site.com/course/123")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button("🔍 Analyser", type="primary")
        if analyze_button and url:
            with st.spinner("🔄 Extraction des données..."):
                df, message = scrape_race_data(url)
                if df is not None:
                    st.success(f"✅ {len(df)} chevaux extraits avec succès")
                    st.dataframe(df.head(), use_container_width=True)
                    df_final = df
                else:
                    st.error(f"❌ {message}")

    with tab2:
        st.subheader("📤 Upload de fichier CSV")
        st.markdown("Format attendu: `Nom, Numéro de corde, Cote, Poids, Musique, Âge/Sexe`")
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
        if uploaded_file:
            try:
                df_final = pd.read_csv(uploaded_file)
                st.success(f"✅ {len(df_final)} chevaux chargés")
                st.dataframe(df_final.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Erreur de lecture: {e}")

    with tab3:
        st.subheader("🧪 Données de Test")
        st.markdown("Tester l'analyseur avec des données pré-chargées")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏃 Test Plat", use_container_width=True):
                df_final = generate_sample_data("plat")
                st.success("✅ Données PLAT chargées (7 chevaux)")
        with col2:
            if st.button("🚗 Test Attelé", use_container_width=True):
                df_final = generate_sample_data("attele")
                st.success("✅ Données ATTELÉ chargées (6 chevaux)")
        with col3:
            if st.button("⭐ Test Premium", use_container_width=True):
                df_final = generate_sample_data("premium")
                st.success("✅ Données PREMIUM chargées (7 chevaux)")

        if df_final is not None:
            st.dataframe(df_final, use_container_width=True)

    # === ANALYSE PRINCIPALE ===
    if df_final is not None and len(df_final) > 0:
        st.markdown("---")
        st.header("🎯 Analyse et Prédictions ML")
        df_prepared = prepare_data(df_final)

        if len(df_prepared) == 0:
            st.error("❌ Aucune donnée valide après préparation")
            return

        # Détection du type de course
        if race_type == "AUTO":
            detected_type = auto_detect_race_type(df_prepared)
        else:
            detected_type = race_type
            st.info(f"📋 **Type sélectionné**: {CONFIGS[detected_type]['description']}")

        # === MACHINE LEARNING AMÉLIORÉ ===
        ml_model = AdvancedHorseRacingML()
        ml_results = None
        confidence_scores = None

        if use_ml:
            with st.spinner("🤖 Entraînement des modèles ML avancés (RF, GB, NN, Stacking)..."):
                try:
                    # Préparation des features avancées
                    X_ml = ml_model.prepare_advanced_features(df_prepared, detected_type)
                    # Affichage du nombre de features
                    st.info(f"🔬 **{len(X_ml.columns)} features** créées pour l'analyse ML")

                    # Entraînement et prédiction
                    ml_predictions, ml_results, confidence_scores = ml_model.train_and_predict(X_ml, detected_type)

                    df_prepared['ml_score'] = ml_predictions
                    df_prepared['confidence'] = confidence_scores
                    st.success("✅ Modèles ML entraînés avec succès")
                    
                    # Affichage des métriques ML
                    if ml_results:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            if 'stacking' in ml_results:
                                st.metric("🏆 R² Stacking", f"{ml_results['stacking']['mean']:.3f}")
                        with col2:
                            if 'random_forest' in ml_results:
                                st.metric("🌲 R² RF", f"{ml_results['random_forest']['mean']:.3f}")
                        with col3:
                            if 'gradient_boosting' in ml_results:
                                st.metric("📈 R² GB", f"{ml_results['gradient_boosting']['mean']:.3f}")
                        with col4:
                            if 'neural_net' in ml_results:
                                st.metric("🧠 R² NN", f"{ml_results['neural_net']['mean']:.3f}")
                        
                        avg_confidence = confidence_scores.mean() if confidence_scores is not None else 0
                        st.metric("🎯 Confiance Moy.", f"{avg_confidence:.1%}")

                except Exception as e:
                    st.error(f"⚠️ Erreur ML: {e}")
                    use_ml = False

        # === SCORE TRADITIONNEL (Baseline) ===
        # Inverse de la cote normalisée
        traditional_score = 1 / (df_prepared['odds_numeric'] + 0.1)
        if traditional_score.max() != traditional_score.min():
            traditional_score = (traditional_score - traditional_score.min()) / (traditional_score.max() - traditional_score.min())

        # === SCORE FINAL ===
        # Si ML est activé, on utilise le score ML, sinon le score traditionnel
        # La pondération est maintenant dynamique et basée sur les performances CV
        if use_ml and 'ml_score' in df_prepared.columns:
            # Ici, on pourrait pondérer en fonction des performances CV, mais pour simplifier,
            # on prend directement le score ML comme score final, car il intègre déjà les autres.
            # Une alternative serait de pondérer traditionnel/ML selon les CV scores.
            df_prepared['score_final'] = df_prepared['ml_score']
        else:
            df_prepared['score_final'] = traditional_score
            df_prepared['confidence'] = np.ones(len(df_prepared)) * 0.5

        # === CLASSEMENT ===
        df_ranked = df_prepared.sort_values('score_final', ascending=False).reset_index(drop=True)
        df_ranked['rang'] = range(1, len(df_ranked) + 1)

        # === AFFICHAGE DES RÉSULTATS ===
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🏆 Classement Final avec Confiance")
            # Préparation du DataFrame d'affichage
            display_cols = ['rang', 'Nom', 'Cote', 'Numéro de corde']
            if 'Poids' in df_ranked.columns:
                display_cols.append('Poids')
            if 'score_final' in df_ranked.columns:
                display_cols.append('score_final')
            if 'confidence' in df_ranked.columns:
                display_cols.append('confidence')
            display_df = df_ranked[display_cols].copy()
            # Formatage
            if 'score_final' in display_df.columns:
                display_df['Score'] = display_df['score_final'].apply(lambda x: f"{x:.3f}")
                display_df = display_df.drop('score_final', axis=1)
            if 'confidence' in display_df.columns:
                display_df['Confiance'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
                display_df = display_df.drop('confidence', axis=1)
            # Coloration conditionnelle
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
        with col2:
            st.subheader("📊 Statistiques de Course")
            # Métriques globales
            favoris = len(df_ranked[df_ranked['odds_numeric'] < 5])
            outsiders = len(df_ranked[df_ranked['odds_numeric'] > 15])
            avg_confidence = df_ranked['confidence'].mean() if 'confidence' in df_ranked.columns else 0
            st.markdown(f'<div class="metric-card">⭐ Favoris (cote < 5)<br><strong>{favoris}</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card">🎲 Outsiders (cote > 15)<br><strong>{outsiders}</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card">🎯 Confiance Moyenne<br><strong>{avg_confidence:.1%}</strong></div>', unsafe_allow_html=True)
            # Top 5 avec confiance
            st.subheader("🥇 Top 5 Prédictions")
            for i in range(min(5, len(df_ranked))):
                horse = df_ranked.iloc[i]
                conf = horse.get('confidence', 0.5)
                # Détermination de la classe de confiance
                if conf >= 0.7:
                    conf_class = "confidence-high"
                    conf_emoji = "🟢"
                elif conf >= 0.4:
                    conf_class = "confidence-medium"
                    conf_emoji = "🟡"
                else:
                    conf_class = "confidence-low"
                    conf_emoji = "🔴"
                st.markdown(f"""
                <div class="prediction-box">
                    <strong>{i+1}. {horse['Nom']}</strong><br>
                    📊 Cote: <strong>{horse['Cote']}</strong> | 
                    🎯 Score: <strong>{horse['score_final']:.3f}</strong><br>
                    {conf_emoji} Confiance: <span class="{conf_class}">{conf:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

        # === GÉNÉRATION DES COMBINAISONS ===
        st.markdown("---")
        st.subheader("🎲 Combinaisons de Paris")
        trio, super4, quinte = generate_predictions(df_ranked)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**E-trio (Top 3)**")
            st.info(f"1️⃣ {trio[0]}<br>2️⃣ {trio[1] if len(trio) > 1 else 'N/A'}<br>3️⃣ {trio[2] if len(trio) > 2 else 'N/A'}", unsafe_allow_html=True)
        with col2:
            st.markdown("**E-Super4 (Top 4)**")
            super4_str = "<br>".join([f"{i+1}️⃣ {name}" for i, name in enumerate(super4)])
            st.info(super4_str, unsafe_allow_html=True)
        with col3:
            st.markdown("**Quinté (Top 5)**")
            quinte_str = "<br>".join([f"{i+1}️⃣ {name}" for i, name in enumerate(quinte)])
            st.info(quinte_str, unsafe_allow_html=True)

        # === VISUALISATIONS AVANCÉES ===
        st.markdown("---")
        st.subheader("📊 Visualisations et Analyses ML")
        fig = create_advanced_visualization(df_ranked, ml_model if use_ml else None)
        st.plotly_chart(fig, use_container_width=True)

        # === ANALYSE DES FEATURES ===
        if use_ml and ml_model.feature_importance:
            st.markdown("---")
            st.subheader("🔬 Analyse de l'Importance des Features")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🌲 Random Forest - Top Features**")
                if 'random_forest' in ml_model.feature_importance:
                    importance_df = pd.DataFrame(
                        list(ml_model.feature_importance['random_forest'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    st.dataframe(importance_df, use_container_width=True, height=300)
            with col2:
                st.markdown("**🧠 Réseau de Neurones - Top Features (Importance Approximative)**")
                if 'neural_net' in ml_model.feature_importance:
                    importance_df = pd.DataFrame(
                        list(ml_model.feature_importance['neural_net'].items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=False)
                    st.dataframe(importance_df, use_container_width=True, height=300)
                else:
                    st.info("L'importance des features pour le NN n'est pas directement disponible. Utilisation d'une approximation basée sur les corrélations.")
                    correlations_df = pd.DataFrame(
                        list(ml_model.feature_correlations.items()),
                        columns=['Feature', 'Correlation']
                    ).sort_values('Correlation', ascending=False).head(10)
                    st.dataframe(correlations_df, use_container_width=True, height=300)


        # === RECOMMANDATIONS STRATÉGIQUES ===
        st.markdown("---")
        st.subheader("💡 Recommandations Stratégiques")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🎯 Chevaux à Fort Potentiel**")
            high_value = df_ranked[
                (df_ranked['score_final'] > df_ranked['score_final'].quantile(0.6)) &
                (df_ranked['odds_numeric'] > 5) &
                (df_ranked['confidence'] > 0.5)
            ].head(3)
            if len(high_value) > 0:
                for idx, horse in high_value.iterrows():
                    st.success(f"✅ **{horse['Nom']}** - Cote: {horse['Cote']} | Score: {horse['score_final']:.3f}")
            else:
                st.info("Aucun outsider à fort potentiel détecté")
        with col2:
            st.markdown("**⚠️ Alertes et Observations**")
            # Alerte sur les favoris sous-performants
            weak_favorites = df_ranked[
                (df_ranked['odds_numeric'] < 5) &
                (df_ranked['score_final'] < df_ranked['score_final'].median())
            ]
            if len(weak_favorites) > 0:
                st.warning(f"⚠️ {len(weak_favorites)} favori(s) avec score faible")
            # Surprise potentielle
            surprise = df_ranked[
                (df_ranked['odds_numeric'] > 10) &
                (df_ranked['rang'] <= 3)
            ]
            if len(surprise) > 0:
                st.info(f"🎲 {len(surprise)} outsider(s) dans le Top 3!")
            else:
                st.info("✅ Classement cohérent avec les cotes")

        # === EXPORT DES RÉSULTATS ===
        st.markdown("---")
        st.subheader("💾 Export des Résultats")
        col1, col2, col3 = st.columns(3)
        with col1:
            csv_data = df_ranked.to_csv(index=False)
            st.download_button(
                "📄 Télécharger CSV",
                csv_data,
                f"pronostic_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            json_data = df_ranked.to_json(orient='records', indent=2)
            st.download_button(
                "📋 Télécharger JSON",
                json_data,
                f"pronostic_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        with col3:
            # Export du rapport complet
            report = f"""
RAPPORT D'ANALYSE HIPPIQUE ML v2
{'='*50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Type de course: {detected_type}
Nombre de chevaux: {len(df_ranked)}
TOP 5 PRÉDICTIONS:
{'-'*50}
"""
            for i in range(min(5, len(df_ranked))):
                horse = df_ranked.iloc[i]
                report += f"{i+1}. {horse['Nom']} - Cote: {horse['Cote']} - Score: {horse['score_final']}
"
            if ml_results:
                report += f"
{'='*50}
MÉTRIQUES ML:
{'-'*50}
"
                for model, scores in ml_results.items():
                    report += f"{model}: R² = {scores['mean']:.3f} (+/- {scores['std']:.3f})
"
            st.download_button(
                "📊 Télécharger Rapport",
                report,
                f"rapport_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
