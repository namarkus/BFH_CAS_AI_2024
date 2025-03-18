#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _digital_marketing_env.py ]________________________________________________________________
"""
Torchrl-Environment für das Digitale Marketing.

Diese Datei enthält die Umgebung für das digitale Marketing, die von der Agentenklasse verwendet
wird, um die Interaktion mit der Umgebung zu steuern.

Das Environment stellt typische Attribute eines Envirnments für das digitale Marketing bereit. Die
Attribute werden dabei simuliert, es wird aber versucht möglichst realistische Werte und Wertfolgen
zu verwenden.

Folgendes wird benötigt:

- adwords.cav: Steuerunggsdatei, welche die zu monitorenden Adwords beinhaltet. Zudem wird pro
  Adword die Competitiveness hinterlegt, wobei 0 dafür steht, dass keine Konkurrenz besteht und 100
  dafür, dass die Konkurrenz sehr stark ist.

Wir haben zwei Interaktionen, welche die Umgebung unabhängig vom Agent beeinflussen:

- add_new_keyword_generation: Ergänzt eine neue Generation der Addwords mit
  allen ihren Attributen in die Umgebung. Alte Generationen werden dabei
  eliminiert.
- register_site_visits: Registriert einen Besucht auf der Website inkl. Herkunft
und Outcome."""

# _____[ Imports ]__________________________________________________________________________________
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import torch

from torchrl.envs import EnvBase
from torchrl.data import Composite, OneHot, Unbounded
from tensordict import TensorDict

import _logging as log
import _metrics as metrics

# _____[ Dataclasses ]__________________________________________________________
@dataclass
class SiteVisit:
    """_summary_
    Dataclass, welche die relevanten Attribute eines Besuchts auf der Website
    definiert.
    """
    url: str
    referer: Optional[str]
    keyword: Optional[str]
    revenue: float

# _____[ Constants ]________________________________________________________________________________
KEYWORD_FIELD = "Keyword"
GENERATION_FIELD = "Generation"
COMPETITIVENESS_FIELD = "Competitiveness"
DIFFICULTY_SCORE_FIELD = "DifficultyScore"
ORGANIC_RANK_FIELD = "OrganicRank"
ORGANIC_CLICKS_FIELD = "OrganicClicks"
ORGANIC_CONVERSIONS_FIELD = "OrganicConversions"
ORGANIC_CLICK_THROUGH_RATE_FIELD = "OrganicClickThroughRate"
AD_CONTINGENT_FIELD = "AdContingent"
AD_IMPRESSION_SHARE_FIELD = "AdImpressionShare"
COST_PER_CLICK_FIELD = "CostPerClick"
AD_SPENT_FIELD = "AdSpent"
AD_CLICKS_FIELD = "AdClicks"
AD_CONVERSIONS_FIELD = "AdConversions"
AD_CLICK_THROUGH_RATE_FIELD = "AdClickThroughRate"
RETURN_ON_AD_SPENT_FIELD = "ReturnOnAdSpent"
OTHER_CLICKS_FIELD = "OtherClicks"
OTHER_CONVERSIONS_FIELD = "OtherConversions"
OTHER_CLICK_THROUGH_RATE_FIELD = "OtherClickThroughRate"
CONVERSION_RATE_FIELD = "ConversionRate"
COST_PER_ACQUISITION_FIELD = "CostPerAcquisition"
MONITOR_FIELD = "Monitor"
TOTAL_CLICKS_FIELD = "TotalClicks"
TOTAL_CONVERSIONS_FIELD = "TotalConversions"

FEATURE_COLUMNS = [COMPETITIVENESS_FIELD, DIFFICULTY_SCORE_FIELD, ORGANIC_RANK_FIELD,
                   ORGANIC_CLICKS_FIELD, ORGANIC_CLICK_THROUGH_RATE_FIELD, AD_CONTINGENT_FIELD,
                   AD_IMPRESSION_SHARE_FIELD, COST_PER_CLICK_FIELD, AD_CLICKS_FIELD, AD_SPENT_FIELD,
                   AD_CLICK_THROUGH_RATE_FIELD, RETURN_ON_AD_SPENT_FIELD, CONVERSION_RATE_FIELD]

DEFAULT_SEED = 42

# _____[ Klasse DigitalMarketingEnv ]_______________________________________________________________
class DigitalMarketingEnv(EnvBase):
    """
    Environment für die Simulation von digitalem Marketing.
    """
    def __init__(self, budget=10_000.00, maximum_class_size=25, campaign_duration_in_days=30, volatility=0.1, adwords_file="adwords.csv"):
        """
        Initialisiert die Umgebung für das digitale Marketing.

        Args:
            budget (_type_, optional): Budget, das für die Kampagne bereitsteht. Defaults to 10_000.00.
            maximum_class_size (int, optional): Grösse der Schulklasse. Defaults to 25.
            campaign_duration_in_days (int, optional): Länge der Werbekampagne in Tagen. Defaults to 30.
            volatility (float, optional): Grad in welchem Random-Werte zwischen zwei Generationen maximal geändert werden darf. Defaults to 0.1.
            adwords_file (str, optional): Dateiname der AddWord-Datei. Defaults to "adwords.csv".
        """
        super().__init__(batch_size=torch.Size([]))
        self.budget = budget
        self.maximum_class_size = maximum_class_size
        self.generations_per_epic = 24 * campaign_duration_in_days
        self.spent_amount = 0.0
        self.reserved_amount = 0.0
        self.current_keyword = None
        self.course_bookings = 0
        self.data_builder = DigitalMarketingSimulation(self, volatility, adwords_file)
        self.keyword_count = len(self.data_builder.persistent_data)
        self._define_specs()

    def __del__(self):
        """
        Beendet die Umgebung (technisch).
        """
        self.stop()

    def stop(self):
        """
        Beendet die Umgebung (fachlich).
        """
        self.data_builder.stop()

    def get_maximum_samples(self):
        """
        Liefert die maximale Anzahl an Samples, die in einer Episode generiert werden können.
        """
        return self.generations_per_epic * self.keyword_count

    def _define_specs(self):
        """
        Definiert die Spezifikationen für TorchRL.

        (!) Im Gegensatz zur ursprünglichen Variante ist hier der Action-Space auf 10 Elemente
            erweitert. Der Agent kann somit 0-9 Inserate zu jedem Wort kaufen.
        """
        self.num_features = len(FEATURE_COLUMNS)
        self.action_spec = Composite(action=OneHot(n=10, dtype=torch.int64))
        self.observation_spec = Composite(observation=Unbounded(shape=(self.num_features,), dtype=torch.float32))
        self.reward_spec = Composite(reward=Unbounded(shape=(1,), dtype=torch.float32))

    def _next_rolling_sample(self):
        """
        Liefert das nächste Sample aus der Generation zurück. Ist keines mehr vorhanden, wird eine
        neue Generation erstellt.
        """
        if not hasattr(self, 'data') or self.data.empty:
            self.generation, self.data = self.data_builder.build_generation()
        sample = self.data.iloc[0]
        self.data = self.data.iloc[1:].reset_index(drop=True)
        self.current_keyword = sample[KEYWORD_FIELD]
        return sample

    def _apply_action(self, action: int):
        """
        Registiert einen eventuellen Kauf.
        """
        self.data_builder.register_ad_reservation(self.current_keyword, action)

    def _reset(self, tensordict=None):
        """
        Setzt die Umgebung zurück.
        """
        log.app_logger().info("🔄 Reset der Umgebung.")
        self.spent_amount = 0.0
        self.reserved_amount = 0.0
        self.course_bookings = 0
        self.data_builder.restart()
        sample = self._next_rolling_sample()
        #print(sample)
        state = torch.tensor(sample[FEATURE_COLUMNS].values.astype(np.float32), dtype=torch.float32)
        return TensorDict({
            "observation": state,
            "done": torch.tensor([False], dtype=torch.bool),  # Explicit shape [1]
        }, batch_size=[])

    def _step(self, tensordict):
        """
        Führt einen Schritt in der Umgebung aus.
        """
        action = tensordict["action"].argmax(dim=-1).item()
        self._apply_action(action)
        next_sample = self._next_rolling_sample()
        #print(next_sample)
        next_state = torch.tensor(next_sample[FEATURE_COLUMNS].values.astype(np.float32), dtype=torch.float32)
        reward = self._compute_reward(action, next_sample)
        terminated = self._check_if_terminated(next_sample)
        truncated = self._check_if_truncated(next_sample)
        done = terminated or truncated

        return TensorDict({
            "observation": next_state,
            "reward": torch.tensor([reward], dtype=torch.float32),
            "done": torch.tensor([done], dtype=torch.bool),
            "terminated": torch.tensor([terminated], dtype=torch.bool),
            "truncated": torch.tensor([truncated], dtype=torch.bool)
            })

    def _compute_reward(self, action, sample:torch.tensor):
        """
        Berechnet den Reward für die Aktion.
        """
        is_buy = action != 0
        bought_ads = action

        overall_success = self.course_bookings # Ganzzahl 0-25
        available_budget = 100 / self.budget * self.spent_amount
        overall_success += available_budget # Bonus für verfügbares Budget in %
        if self.course_bookings >= self.maximum_class_size:
            overall_success += 100 # Bonus für erfolgreichen Abschuss
        if self.spent_amount > self.budget:
            overall_success -= 100  # Penalty für Budgetüberschreitung
        if self.data_builder.current_generation >= self.generations_per_epic:
            overall_success -= 100 # Penalty für Kampagnenabbruch

        difficulty = sample[DIFFICULTY_SCORE_FIELD]
        organic_rank = sample[ORGANIC_RANK_FIELD]
        is_search_prefered = organic_rank <= 5

        difficulty_reward = (10 - difficulty) / 10
        if not is_buy and not is_search_prefered:
            difficulty_reward *= -1

        if is_buy and is_search_prefered:
            # Kauf, obwohl in den Suchergebnissen gut platziert
            organic_rank_penalty = (5 - organic_rank) * bought_ads * -1
        else:
            organic_rank_penalty = 0

        ad_continget = sample[AD_CONTINGENT_FIELD]
        if is_buy and ad_continget > 0:
            # Kauf, obwohl noch nicht alle Inserate verkauft sind.
            ad_hoarding_penalty = ad_continget * bought_ads * -1
        else:
            ad_hoarding_penalty = 0

        iteration_count_penalty = self.data_builder.current_generation * -0.1

        reward = overall_success + difficulty_reward + organic_rank_penalty + ad_hoarding_penalty + iteration_count_penalty
        self.data_builder.log_reward(self.current_keyword, reward) # prüfen: müsste hier nicht das Keyword von previous sample genommen werden?

        #if self.data_builder.current_generation % 10 == 0:
        #    log.app_logger().info(f"🎯 Reward: {reward} for action {action}: (overall_success: {overall_success}, difficulty_reward: {difficulty_reward}, organic_rank_penalty: {organic_rank_penalty}, ad_hoarding_penalty: {ad_hoarding_penalty}, iteration_count_penalty: {iteration_count_penalty})")

        return reward

    def _check_if_truncated(self, sample):
        """
        Prüft, ob die Episode abgebrochen werden muss. Dies ist der Fall, wenn der zeitlich gesetzte
        Rahmen der Kampagne erreicht ist.
        """
        if self.data_builder.current_generation >= self.generations_per_epic:
            log.app_logger().info(f"⌛️ Abbruch (truncated), da Kampagnendurchlaufzeit beendet nach {self.data_builder.current_generation} Generationen. Ausgegeben wurden CHF {self.spent_amount} vom Budget von CHF {self.budget}.Bisher sind {self.course_bookings} Kursbuchungen erfolgt.")
            return True
        return False

    def _check_if_terminated(self, sample):
        """
        Prüft, ob die Episode beendet werden soll. Dies ist der Fall, wenn
        - das Budget aufgebraucht ist (verloren)
        - alle Kursplätze belegt sind (gewonnen)
        """
        if self.course_bookings >= self.maximum_class_size:
            log.app_logger().info(f"👍🏻 Erfolgreicher Abschluss (terminated), da {self.course_bookings} Kursplätze belegt sind. nach {self.data_builder.current_generation} Generationen. Vom Budget von von CHF {self.budget} wurden CHF {self.spent_amount} ausgegeben.")
            return True
        if self.spent_amount > self.budget:
            log.app_logger().info(f"👎🏻 Erfolgloser Abschluss (terminated), da das Budget von CHF {self.budget} aufgebraucht ist. Es wurden {self.course_bookings} Kurse gebucht.")
            return True
        return False

    def _set_seed(self, seed: Optional[int]):
        """
        Setzt den Seed für die Umgebung. Somiot kann die Wiederholbarkeit der Simulation
        sichergestellt werden.
        """
        rng = torch.manual_seed(seed)
        self.rng = rng


 # _____[ Klasse DigitalMarketingSimulation ]_______________________________________________________
class DigitalMarketingSimulation:
    """
    Simuliert die Daten für das digitale Marketing.
    """
    def __init__(self, env: DigitalMarketingEnv, volatility: float, adwords_file: str):
        """
        Initialisiert die Simulation für das digitale Marketing.
        params:
        - env: Referenz auf das Environment
        - volatility: Grad der Sprunghaftigkeit der Daten
        - adwords_file: Dateiname der AdWords-Datei
        """
        self.env = env
        self.volatility = volatility
        self.metrics = metrics.TensorBoardMonitor.instance()
        self.current_step = 0
        self.random = np.random.default_rng(DEFAULT_SEED)
        self.adwords_file = adwords_file
        self.restart()
        _, self.current_marketing_data = self.build_generation()
        self.site_analytics_simulation = SiteAnalyticsSimulation(self, self.persistent_data[KEYWORD_FIELD])
        self.site_analytics_simulation.start()

    def restart(self):
        """
        Startet die Simulation neu. Dies kann z.B. beim Neuaufbau der Umgebung oder beim Start einer
        neuen Episode notwendig sein.
        """
        self.current_generation = 0
        self.revenue = 0.0
        self.persistent_data = pd.read_csv(self.adwords_file, sep=";", header=0, encoding="utf-8")
        #print(self.persistent_data)

    def __del__(self):
        """
        Beendet die Simulation (technischer Aufruf).
        """
        self.stop()

    def stop(self):
        """
        Beendet die Simulation (fachlicher Aufruf).
        """
        self.site_analytics_simulation.stop()
        del self.site_analytics_simulation

    def build_generation(self) -> Tuple[int, pd.DataFrame]:
        """
        Erstellt eine neue Generation von Marketing-Daten.
        """
        self.current_generation += 1
        self.current_step += 1
        self.update_volatile_peristent_data()
        generation_data = self.persistent_data.copy()
        all_searches = generation_data[ORGANIC_CLICKS_FIELD].sum()
        generation_data[DIFFICULTY_SCORE_FIELD] = generation_data.apply(lambda row: self._calculate_difficulty_score(row[COMPETITIVENESS_FIELD], row[ORGANIC_CLICKS_FIELD], all_searches), axis=1)
        generation_data[ORGANIC_RANK_FIELD] = generation_data[COMPETITIVENESS_FIELD].apply(self._calculate_organic_rank)
        generation_data[ORGANIC_CLICK_THROUGH_RATE_FIELD] = generation_data.apply(lambda row: self._calculate_ctr(row[ORGANIC_CLICKS_FIELD], row[ORGANIC_CONVERSIONS_FIELD]), axis=1)
        generation_data[AD_IMPRESSION_SHARE_FIELD] = generation_data.apply(lambda row: self._calculate_impression_share(row[AD_CONTINGENT_FIELD], row[COMPETITIVENESS_FIELD]), axis=1)
        generation_data[AD_CLICK_THROUGH_RATE_FIELD] = generation_data.apply(lambda row: self._calculate_ctr(row[AD_CLICKS_FIELD], row[AD_CONVERSIONS_FIELD]), axis=1)
        generation_data[RETURN_ON_AD_SPENT_FIELD] = generation_data.apply(lambda row: self._calculate_return_on_ad_spent(row[AD_SPENT_FIELD], row[AD_CONVERSIONS_FIELD]),axis=1)
        generation_data[OTHER_CLICK_THROUGH_RATE_FIELD] = generation_data.apply(lambda row:self._calculate_ctr(row[OTHER_CLICKS_FIELD], row[OTHER_CONVERSIONS_FIELD]),axis=1)
        generation_data[TOTAL_CLICKS_FIELD] = generation_data[ORGANIC_CLICKS_FIELD] + generation_data[AD_CLICKS_FIELD] + generation_data[OTHER_CLICKS_FIELD]
        generation_data[TOTAL_CONVERSIONS_FIELD] = generation_data[ORGANIC_CONVERSIONS_FIELD] + generation_data[AD_CONVERSIONS_FIELD] + generation_data[OTHER_CONVERSIONS_FIELD]
        generation_data[CONVERSION_RATE_FIELD] = generation_data.apply(lambda row: self._calculate_conversion_rate(row[TOTAL_CLICKS_FIELD], row[TOTAL_CONVERSIONS_FIELD]), axis=1)
        ad_spent_overall = generation_data[AD_SPENT_FIELD].sum()
        generation_data[COST_PER_ACQUISITION_FIELD] = generation_data[TOTAL_CONVERSIONS_FIELD].apply(self._calculate_cost_per_acquisition, args=[ad_spent_overall])
        #print(generation_data)
        self._add_to_monitor(generation_data[generation_data[MONITOR_FIELD] == 1])
        log.app_logger().info(f"Daten für Generation {self.current_generation} wurden erstellt.")
        return self.current_generation,  generation_data

    def update_volatile_peristent_data(self):
        """
        Aktualisiert die unbeständigen persistenten Daten, indem diese im Rahmen der definierten
        Sprunghaftigkeit zufällig geändert werden.
        """
        self.persistent_data[COMPETITIVENESS_FIELD] = self.persistent_data[COMPETITIVENESS_FIELD].apply(self._clipped_volatile_float, args=[0.00, 1.00])
        self.persistent_data[COST_PER_CLICK_FIELD] = self.persistent_data[COST_PER_CLICK_FIELD].apply(self._clipped_volatile_price)

    def _clipped_volatile_float(self, value: float, min_value: float, max_value: float) -> float:
        """
        Ändert den Wert um einen zufälligen Wert im erlaubten Bereich.
        """
        changed = self.random.uniform(1 - self.volatility, 1 + self.volatility) * value
        return np.clip(changed, min_value, max_value)

    def _clipped_volatile_int(self, value: int, min_value: int, max_value: int) -> int:
        """
        Ändert den Wert um einen zufälligen Wert im erlaubten Bereich.
        """
        changed = (self.random.uniform(1 - self.volatility, 1 + self.volatility) * value).astype(int)
        return np.clip(changed, min_value, max_value)

    def _clipped_volatile_price(self, value: float, min_value=0.20, max_value=4.00) -> float:
        """
        Ändert den Preis eines Inserats um einen zufälligen Wert im erlaubten Bereich.
        params:
        - value: Preis
        - min_value: Minimaler Preis (default: 0.20)
        - max_value: Maximaler Preis (default: 4.00)
        """
        if value < 0.01:
            return self.random.uniform(min_value, max_value)
        changed = self.random.uniform(1 - self.volatility, 1 + self.volatility) * value
        return np.clip(changed, min_value, max_value)

    def _calculate_difficulty_score(self, competitiveness: float, current_search_count: int, all_search_count:int) -> int:
        """
        Berechnet den Schwierigkeitsgrad eines Keywords.
        params:
        - competitiveness: Konkurrenzgrad des Keywords
        - current_search_count: Anzahl der aktuellen Suchen
        - all_search_count: Gesamtanzahl der Suchen
        """
        theoretic_rank = competitiveness * 100
        if all_search_count == 0 or current_search_count == 0:
            real_rank = theoretic_rank
        else:
            real_rank = (all_search_count - current_search_count) / (all_search_count + current_search_count)
        rank = (real_rank + theoretic_rank) / 2
        #print(f"theoretic rank: {theoretic_rank} + real rank ({all_search_count} - {current_search_count}) : {real_rank} / 2 = rank: {rank}")
        return np.round(rank).astype(int)

    def _calculate_organic_rank(self, competitiveness: float) -> int:
        """
        Berechnet den organischen Rang eines Keywords. Auch hier wird die Sprunghaftigkeit
        berücksichtigt, damit sich die Werte zwischen den Generationen in gewissen Bandbreiten ändern.
        params:
        - competitiveness: Konkurrenzgrad des Keywords
        """
        rank = self.random.uniform(1 - self.volatility, 1 + self.volatility) * competitiveness * 100
        return np.round(rank).astype(int)

    def _calculate_impression_share(self, contingent: int, competitiveness: float) -> float:
        """
        Berechnet den Anteil der Impressionen, die ein Keyword erhält.
        params:
        - contingent: Anzahl der verfügbaren Inserate
        - competitiveness: Konkurrenzgrad des Keywords
        """
        if contingent == 0:
            return 0.0
        return self.random.uniform(1 - self.volatility, 1 + self.volatility) * contingent * (1 - competitiveness)

    def _calculate_ctr(self, total_clicks: int, conversion_clicks) -> float:
        """
        Berechnet die Click-Through-Rate (CTR) eines Keywords.
        params:
        - total_clicks: Gesamtanzahl der Klicks
        - conversion_clicks: Anzahl der Klicks, die zu einer Konversion geführt haben
        """
        if total_clicks == 0:
            return 0
        return 100 * conversion_clicks / total_clicks

    def _calculate_return_on_ad_spent(self, ad_spent: float, ad_conversions: int) -> float:
        """
        Berechnet den Return on Ad Spent (ROAS) eines Keywords.
        params:
        - ad_spent: Ausgaben für das Keyword
        - ad_conversions: Anzahl der Konversionen
        """
        if ad_spent < 0.0001 or ad_conversions == 0 or self.revenue == 0:
            return 0
        return 100 / ad_conversions * ad_spent * self.revenue

    def _calculate_conversion_rate(self, total_clicks: int, total_conversions: int) -> float:
        """
        Berechnet die Konversionsrate eines Keywords.
        params:
        - total_clicks: Gesamtanzahl der Klicks
        - total_conversions: Gesamtanzahl der Konversionen
        """
        if total_clicks == 0:
            return 0
        return 100 * total_conversions / total_clicks

    def _calculate_cost_per_acquisition(self, total_conversions: int, ad_spent_overall: float) -> float:
        """
        Berechnet die Kosten pro Akquisition (CPA) eines Keywords.
        params:
        - total_conversions: Gesamtanzahl der Konversionen
        - ad_spent_overall: Gesamtausgaben für das Keyword
        """
        if total_conversions == 0:
            return 0.0
        return ad_spent_overall / total_conversions

    def register_ad_reservation(self, keyword: str, ad_count: int):
        """
        Registriert die Reservierung von Inseraten für ein Keyword.
        params:
        - keyword: Keyword, für das Inserate reserviert werden
        - ad_count: Anzahl der reservierten Inserate
        """
        keyword_index = self.persistent_data[self.persistent_data[KEYWORD_FIELD] == keyword].index[0]
        if keyword_index > -1 and ad_count > 0:
            #print(f'Kaufe {ad_count} Inserate von {keyword} (keyword_index: {keyword_index})')
            self.persistent_data.loc[keyword_index, AD_CONTINGENT_FIELD] += ad_count
            self.env.reserved_amount += ad_count * self.persistent_data.loc[keyword_index, COST_PER_CLICK_FIELD]

    def register_site_visit(self, site_visit: SiteVisit):
        """
        Registriert einen Besuch auf der Website.
        params:
        - site_visit: Besuch auf der Website"""
        keyword_index = self.persistent_data[self.persistent_data[KEYWORD_FIELD] == site_visit.keyword].index[0]
        if keyword_index > -1:
            if site_visit.revenue > 0.0:
                conversion = 1
                self.env.course_bookings += 1
                self.revenue += site_visit.revenue
                log.app_logger().info(f"{site_visit} endete in einer Kursbuchung.")
            else:
                conversion = 0
            if site_visit.referer == "Search":
                self.persistent_data.loc[keyword_index, ORGANIC_CLICKS_FIELD] += 1
                self.persistent_data.loc[keyword_index, ORGANIC_CONVERSIONS_FIELD] += conversion
            if site_visit.referer == "AdWords":
                self.persistent_data.loc[keyword_index, AD_CLICKS_FIELD] += 1
                self.persistent_data.loc[keyword_index, AD_CONTINGENT_FIELD] -= 1
                self.persistent_data.loc[keyword_index, AD_CONVERSIONS_FIELD] += conversion
                effective_cost = self.persistent_data.loc[keyword_index, COST_PER_CLICK_FIELD]
                self.persistent_data.loc[keyword_index, AD_SPENT_FIELD] += effective_cost
                self.env.spent_amount += effective_cost
            if site_visit.referer is None:
                self.persistent_data.loc[keyword_index, OTHER_CLICKS_FIELD] += 1
                self.persistent_data.loc[keyword_index, OTHER_CONVERSIONS_FIELD] += conversion
            #if self.persistent_data.loc[keyword_index, "Monitor"] == 1:
            #    print(self.persistent_data.loc[keyword_index])


    def get_most_likely_search_terms(self):
        """""
        Liefert die Keywords mit den besten organischen Suchergebnissen zurück.
        """
        if self.current_marketing_data is None:
            return pd.Series([])
        return self.current_marketing_data[self.current_marketing_data[ORGANIC_RANK_FIELD] < 10].sort_values(by=ORGANIC_RANK_FIELD, ascending=True)[KEYWORD_FIELD]

    def get_current_terms_with_ad_contingent(self):
        """""
        Liefert alle Suchbegriffe zurück, welche noch ein Ad-Contingent haben. Bei der Reihenfolge
        wird die Competitiveness berücksichtigt. Das relevanteste Keyword wird zuerst zurückgegeben.
        """
        #print(self.current_marketing_data[self.current_marketing_data[AD_CONTINGENT_FIELD] > 0])
        #return self.current_marketing_data[self.current_marketing_data[AD_CONTINGENT_FIELD] > 0].sort_values(by=DIFFICULTY_SCORE_FIELD, ascending=True)[KEYWORD_FIELD]
        return self.persistent_data[self.persistent_data[AD_CONTINGENT_FIELD] > 0].sort_values(by=COMPETITIVENESS_FIELD, ascending=True)[KEYWORD_FIELD]

    def _add_to_monitor(self, data: pd.DataFrame):
        """
        Fügt die Daten für das Monitoring via Tensorboard hinzu.
        params:
        - data: Daten, die für das Monitoring hinzugefügt werden sollen
        """
        self.metrics.log_metrics({
                "0 - Betrag ausgegeben": self.env.spent_amount,
                "0 - Betrag reserviert": self.env.reserved_amount,
                "0 - Buchungen von Kursen": self.env.course_bookings
        }, step=self.current_step)
        for index, row in data.iterrows():
            keyword = row[KEYWORD_FIELD]
            self.metrics.log_metrics({
                f"{COMPETITIVENESS_FIELD} ({keyword})": row[COMPETITIVENESS_FIELD],
                f"{DIFFICULTY_SCORE_FIELD} ({keyword})": row[DIFFICULTY_SCORE_FIELD],
                f"{ORGANIC_RANK_FIELD} ({keyword})": row[ORGANIC_RANK_FIELD],
                f"{ORGANIC_CLICKS_FIELD} ({keyword})": row[ORGANIC_CLICKS_FIELD],
                f"{ORGANIC_CLICK_THROUGH_RATE_FIELD} ({keyword})": row[ORGANIC_CLICK_THROUGH_RATE_FIELD],
                f"{AD_CONTINGENT_FIELD} ({keyword})": row[AD_CONTINGENT_FIELD],
                f"{AD_IMPRESSION_SHARE_FIELD} ({keyword})": row[AD_IMPRESSION_SHARE_FIELD],
                f"{COST_PER_CLICK_FIELD} ({keyword})": row[COST_PER_CLICK_FIELD],
                f"{AD_CLICKS_FIELD} ({keyword})": row[AD_CLICKS_FIELD],
                f"{AD_SPENT_FIELD} ({keyword})": row[AD_SPENT_FIELD],
                f"{AD_CLICK_THROUGH_RATE_FIELD} ({keyword})": row[AD_CLICK_THROUGH_RATE_FIELD],
                f"{RETURN_ON_AD_SPENT_FIELD} ({keyword})": row[RETURN_ON_AD_SPENT_FIELD],
                f"{OTHER_CLICKS_FIELD} ({keyword})": row[OTHER_CLICKS_FIELD],
                f"{OTHER_CONVERSIONS_FIELD} ({keyword})": row[OTHER_CONVERSIONS_FIELD],
                f"{OTHER_CLICK_THROUGH_RATE_FIELD} ({keyword})": row[OTHER_CLICK_THROUGH_RATE_FIELD],
                f"{CONVERSION_RATE_FIELD} ({keyword})": row[CONVERSION_RATE_FIELD],
                f"{TOTAL_CLICKS_FIELD} ({keyword})": row[TOTAL_CLICKS_FIELD],
                f"{TOTAL_CONVERSIONS_FIELD} ({keyword})": row[TOTAL_CONVERSIONS_FIELD]
            }, step=self.current_step)

    def log_reward(self, keyword: str, reward: float):
        """
        Loggt den Reward für ein Keyword.
        params:
        - keyword: Keyword, für das der Reward geloggt werden soll
        - reward: Reward
        """
        monitor =self.persistent_data.loc[self.persistent_data[KEYWORD_FIELD] == keyword][MONITOR_FIELD].values[0]
        if monitor == 1:
            self.metrics.log_metrics({f"Reward ({keyword})": reward}, step=self.current_step)


# _____[ Klasse SiteAnalyticsSimulation ]___________________________________________________________
class SiteAnalyticsSimulation:
    """
    Simuliert die Analyse der  Aktivitäten auf der Website. Random werden Seiten mit verschieden
    Quellen und Keywords aufgerufen. Abschlüsse werden ebenfalls über die Analytics-Simulation
    geliefert.

    todo:
    - Ausbauvariante Landingpages einlesen und über diese Keywords und Kurspreis ermitteln. Im MVP
      wird nur eine fix codierte Landingpage verwendet.
    """
    def __init__(self, target: DigitalMarketingSimulation, keywords: pd.Series):
        """
        Initialisiert die Simulation der Website-Analytik.
        params:
        - target: Referenz auf die digitale Marketing-Simulation
        - keywords: Keywords, die für die Simulation verwendet werden
        """
        self.target = target
        self.keywords = keywords
        self.random = np.random.default_rng(DEFAULT_SEED)
        self.started = False

    def __del__(self):
        """
        Beendet die Simulation (technischer Aufruf).
        """
        self.stop()

    def start(self):
        """
        Startet die Simulation.
        """
        log.app_logger().info("Starte Analytics-Simulation ...")
        self.started = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        """
        Beendet die Simulation.
        """
        if self.thread:
            log.app_logger().info("Stoppe Analytics-Simulation ...")
            self.started = False
            self.thread.join(timeout=0.1)
            if self.thread.is_alive():
                log.app_logger().info("Thread konnte nicht beendet werden. Versuche es erneut.")
                self.stop()
            else:
                self.thread = None

    def run(self):
        """
        Führt die Simulation aus.Diese läuft dabei solante in einer Englosschleife, bis die
        Simulation von der stuernden Klasse aus beendet wird.
        """
        while self.started:
            origin = self.random.choice(["Search", "AdWords", None], p=[1/3, 1/3, 1/3])
            if origin == "AdWords":
                keywords = self.target.get_current_terms_with_ad_contingent()
                if len(keywords) > 0:
                    weights = np.linspace(1, 0.1, len(keywords))  # Höhere Gewichtung für Einträge mit tieferer Competitiveness
                    weights /= weights.sum()
                    keyword = keywords.sample(weights=weights).values[0]
                    revenue=self.random.choice([6500.00, 0.00], p=[0.002, 0.998]) # 1:500
                else:
                    origin = None
            if origin == "Search":
                keywords = self.target.get_most_likely_search_terms()
                weights = np.linspace(1, 0.1, len(keywords))  # Höhere Gewichtung für frühere Einträge
                weights /= weights.sum()
                keyword = keywords.sample(weights=weights).values[0]
                revenue=self.random.choice([6500.00, 0.00], p=[0.001, 0.999]) # 1:1000
            if origin is None:
                keyword=self.keywords.sample().values[0]
                revenue=self.random.choice([6500.00, 0.00], p=[0.0005, 0.9995]) # 1:2000
            site_visit = SiteVisit(
                url="https://www.bfh.ch/de/weiterbildung/cas/artificial-intelligence/",
                referer=origin,
                keyword=keyword,
                revenue=revenue,
            )
            log.app_logger().debug(site_visit)
            self.target.register_site_visit(site_visit)
            user_activity_level = self.random.uniform(0.001, 0.01)
            time.sleep(user_activity_level)


# _____[ Main (Test/Simulation des Envs)]___________________________________________________________
if __name__ == '__main__':
    log.start_logger("DigitalMarketingEnv", "dev")
    log.app_logger().info("Starte Simulation des DigitalMarketingEnv")
    log.app_logger().info("Simuliere Environment-Dynamic eine Minute lang")
    env = DigitalMarketingEnv()
    time.sleep(60.0)
    env.stop()
    log.app_logger().info("Simulation beendet")
