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
from typing import Optional
import pandas as pd
import numpy as np
import torch
import torchrl

import _logging as log
#import _metrics as metrics

# _____[ Dataclasses ]__________________________________________________________
@dataclass
class SiteVisit:
    """_summary_
    Dataclass, welche die relevatnen Attribute eines Besuchts auf der Website  
    definiert.
    """
    url: str
    referer: Optional[str]
    keyword: Optional[str]
    revenue: float

@dataclass
class KeywordValue:
    # Generation des Keywords
    generation : int
    # Das Keyword selbst
    keyword : str
    # Wettbewerbsfähigkeit (je höher, desto schwerer); Wir inital aus adwords.csv vergeben, kann 
    # sich aber im Laufe der Zeit random in gewissen Grenzen verändern.
    competitiveness : float
    difficulty_score : float
    # --- Organische Werte (Suchmaschine) ---
    # Organisches Ranking (1 ist das beste)
    # Nimmt als Basis die Wettbewerbsfähigkeit und wird durch Zufall angepasst, ausser wenn keine 
    # Konkurrenz besteht.
    organic_rank : int
    # Organische Klicks (mehr Klicks bei besserem Rank); Wird anhand der generierten Analytcs-Clicks 
    # gesetzt.
    organic_clicks : int
    # Organische click_through_rate (höherer Rank → höhere click_through_rate)
    # 
    organic_click_through_rate : float
    # --- Bezahlte Werte (AdWords) ---
    # Kontingent der gekauften Werbung.
    # todo: neu: überall noch ergänzen.
    ad_contingent: int
    # Summe aller Ausgaben für Werbung.
    # Wird immer erhöht, wenn eine Werbung geschaltet wird. (oder doch pro Click?)
    ad_spend : float 
    # Anzahl bezahlter Clicks. 
    # Wird anhand der Analytics gesetzt.
    paid_clicks : int
    #  Click-Through-Rate (CTR) ist das Verhältnis von Klicks auf eine Anzeige zu den Impressionen.
    # 
    paid_click_through_rate : float
    # Kosten pro Klick (CPC steigt mit Competitiveness)
    cost_per_click : float
    # Konversionsrate Klick in Relation zu ,,,,
    conversion_rate : float
    # Ad Conversions (abhängig von Paid Clicks und Conversion Rate)
    ad_conversions : int
    ad_roas : float
    cost_per_acquisition : float
    impression_share : float
    previous_recommendation : bool
# _____[ Constants ]________________________________________________________________________________
feature_columns = ["competitiveness", "difficulty_score", "organic_rank", "organic_clicks", 
                   "organic_click_through_rate", "ad_contingent", "paid_clicks", 
                   "paid_click_through_rate", "ad_spend", "ad_conversions", "ad_roas", 
                   "conversion_rate", "cost_per_click"]
    
# _____[ Environment ]_________________________________1_____________________________________________
class DigitalMarketingEnv:
    """_summary_
    Environment für die Simulation von digitalem Marketing. 
    """
    def __init__(self, budget=10_000.00, maximum_class_size=25, volatility=0.1, max_generations=10, adwords_file="adwords.csv"):
        self.currently_used_keywords = []
        self.initial_budget = budget
        self.budget = budget
        self.maximum_class_size = maximum_class_size        
        self.course_bookings = 0
        self.max_generations = max_generations
        self.keyword_values_builder = KeywordValueOfferSimulation(volatility, adwords_file)
        current_generation, self.keyword_values = self.keyword_values_builder.batch_build(10)

    def __del__(self):
        del self.keyword_values_builder    

    def _reset(self, tensordict=None):
        self.budget = self.initial_budget
        sample = self.keyword_values.sample(1)
        state = torch.tensor(sample[feature_columns].values, dtype=torch.float32).squeeze()
        return tensordict.TensorDict({"observation": state}, batch_size=[])

    def _step(self, tensordict):
        action = tensordict["action"].argmax(dim=-1).item()
        next_sample = self.dataset.sample(1)

        next_state = torch.tensor(next_sample[feature_columns].values, dtype=torch.float32).squeeze()
        reward = self._compute_reward(action, next_sample)

        # Determine if the episode is done
        done = self._check_if_done(next_sample)  # Replace with your termination logic

        # Use step_and_maybe_reset to manage resets
        return step_and_maybe_reset(
            env=self,  # Reference to the environment
            tensordict_out=TensorDict({
                "observation": next_state,
                "reward": torch.tensor([reward], dtype=torch.float32),  # Ensure reward is always added
                "done": torch.tensor([done], dtype=torch.bool)
            }, batch_size=[1]),
            done_key="done"
        )


    def _compute_reward(self, action, sample):
        if action != 0: # Kaufen
            is_buy = True
        else:
            is_buy = False

        # könnte evtl. verwirren, da nicht direkt im Zusammenhang mit der letzten Aktion.
        overall_success = self.course_bookings

        if is_buy and sample.organic_rank < 5:
            # Kauf, obwohl in den Suchergebnissen gut platziert
            organic_rank_penalty = (sample.organic_rank - 4) * -1
        else:
            organic_rank_penalty = 0

        if is_buy and sample.ad_contingent > 0:
            # Kauf, obwohl noch nicht alle Inserate verkauf sind.
            ad_hoarding_penalty = sample.ad_contingent * -1
        else:
            ad_hoarding_penalty = 0
        
        # ...

        return overall_success + organic_rank_penalty + ad_hoarding_penalty

    def _check_if_done(self, sample):
        if self.budget <= 0:
            return True
        if self.course_bookings >= self.maximum_class_size:
            return True
        return False

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

 
class KeywordValueOfferSimulation:
    def __init__(self, volatility: float, adwords_file: str):
        self.current_generation = 0        
        self.volatility = volatility
        self.keywords = pd.read_csv(adwords_file, sep=";", header=0, encoding="utf-8")
        print(self.keywords)
        self.site_analytics_simulation = SiteAnalyticsSimulation(self, self.keywords["Keyword"])
        self.site_analytics_simulation.start()
        self.last_created_generation = self._create_inital_generation()
    
    def __del__(self):
        del self.site_analytics_simulation

    def _create_inital_generation(self) -> pd.DataFrame:
        initial_generation = []
        for keyword_index in self.keywords.index:
            keyword = self.keywords.loc[keyword_index, "Keyword"]
            competitors = self.keywords.loc[keyword_index, "Competitors"]
            competitiveness = self._revive_competitiveness(competitors)
            difficulty_score = self._randomize_dificulty_score(competitiveness)
            organic_rank = (competitors * np.random.randint(1, 10)) + 1
            organic_clicks = self.keywords.loc[keyword_index, "SearchClicks"]
            organic_click_through_rate = self._calculate_ctr(organic_clicks, self.keywords.loc[keyword_index, "SearchConversions"])
            ad_spend = self.keywords.loc[keyword_index, "AdSpent"]
            paid_clicks = self.keywords.loc[keyword_index, "AdClicks"]
            add_conversions = self.keywords.loc[keyword_index, "AdConversions"]
            paid_click_through_rate = self._calculate_ctr(paid_clicks, add_conversions)
            cost_per_click = np.clip(np.random.normal(competitiveness * 10, 1), 0.1, 10)    
            keyword_value = KeywordValue(
                generation=self.current_generation,
                keyword=keyword,
                competitiveness=competitiveness,
                difficulty_score=difficulty_score,
                organic_rank=organic_rank,
                organic_clicks=organic_clicks,
                organic_click_through_rate=organic_click_through_rate,
                ad_spend=ad_spend,
                paid_clicks=paid_clicks,
                paid_click_through_rate=paid_click_through_rate,
                cost_per_click=cost_per_click,
                conversion_rate=0.01,
                ad_conversions=add_conversions,
                ad_roas=0.0,
                cost_per_acquisition=0.0,
                impression_share=0.0,
                previous_recommendation=False,
            )
            initial_generation.append(keyword_value)
            if self.keywords.loc[keyword_index, "Monitor"] == 1:
                self._log(keyword_value)
        self.current_generation += 1    
        return pd.DataFrame(initial_generation)

    def _create_inital_generation(self) -> pd.DataFrame:
        initial_generation = []
        for keyword_index in self.keywords.index:
            keyword = self.keywords.loc[keyword_index, "Keyword"]
            competitors = self.keywords.loc[keyword_index, "Competitors"]
            competitiveness = self._revive_competitiveness(competitors)
            difficulty_score = self._randomize_dificulty_score(competitiveness)
            organic_rank = (competitors * np.random.randint(1, 10)) + 1
            organic_clicks = self.keywords.loc[keyword_index, "SearchClicks"]
            organic_click_through_rate = self._calculate_ctr(organic_clicks, self.keywords.loc[keyword_index, "SearchConversions"])
            ad_contingent = self.keywords.loc[keyword_index, "AdContingent"]
            ad_spend = self.keywords.loc[keyword_index, "AdSpent"]
            paid_clicks = self.keywords.loc[keyword_index, "AdClicks"]
            add_conversions = self.keywords.loc[keyword_index, "AdConversions"]
            paid_click_through_rate = self._calculate_ctr(paid_clicks, add_conversions)
            cost_per_click = np.clip(np.random.normal(competitiveness * 10, 1), 0.1, 10)    
            keyword_value = KeywordValue(
                generation=self.current_generation,
                keyword=keyword,
                competitiveness=competitiveness,
                difficulty_score=difficulty_score,
                organic_rank=organic_rank,
                organic_clicks=organic_clicks,
                organic_click_through_rate=organic_click_through_rate,
                ad_contingent=ad_contingent,
                ad_spend=ad_spend,
                paid_clicks=paid_clicks,
                paid_click_through_rate=paid_click_through_rate,
                cost_per_click=cost_per_click,
                conversion_rate=0.01,
                ad_conversions=add_conversions,
                ad_roas=0.0,
                cost_per_acquisition=0.0,
                impression_share=0.0,
                previous_recommendation=False,
            )
            initial_generation.append(keyword_value)
            if self.keywords.loc[keyword_index, "Monitor"] == 1:
                self._log(keyword_value)
        self.current_generation += 1    
        return pd.DataFrame(initial_generation)

    def build_generation(self) -> pd.DataFrame:
        current_generation = []
        for keyword_index in self.keywords.index:
            keyword = self.keywords.loc[keyword_index, "Keyword"]
            competitors = self.keywords.loc[keyword_index, "Competitors"]
            competitiveness = self._revive_volatile_value(self.keywords.loc[keyword_index, "Competitors"])
            difficulty_score = self._randomize_dificulty_score(competitiveness)
            organic_rank = (competitors * np.random.randint(1, 10)) + 1
            organic_clicks = self.keywords.loc[keyword_index, "SearchClicks"]
            organic_click_through_rate = self._calculate_ctr(organic_clicks, self.keywords.loc[keyword_index, "SearchConversions"])
            ad_contingent=self.keywords.loc[keyword_index, "AdContingent"]
            ad_spend = self.keywords.loc[keyword_index, "AdSpent"]
            paid_clicks = self.keywords.loc[keyword_index, "AdClicks"]
            add_conversions = self.keywords.loc[keyword_index, "AdConversions"]
            paid_click_through_rate = self._calculate_ctr(paid_clicks, add_conversions)
            cost_per_click = np.clip(np.random.normal(competitiveness * 10, 1), 0.1, 10)    
            keyword_value = KeywordValue(
                generation=self.current_generation,
                keyword=keyword,
                competitiveness=competitiveness,
                difficulty_score=difficulty_score,
                organic_rank=organic_rank,
                organic_clicks=organic_clicks,
                organic_click_through_rate=organic_click_through_rate,
                ad_contingent=ad_contingent,
                ad_spend=ad_spend,
                paid_clicks=paid_clicks,
                paid_click_through_rate=paid_click_through_rate,
                cost_per_click=cost_per_click,
                conversion_rate=0.01,
                ad_conversions=add_conversions,
                ad_roas=0.0,
                cost_per_acquisition=0.0,
                impression_share=0.0,
                previous_recommendation=False,
            )
            current_generation.append(keyword_value)
            if self.keywords.loc[keyword_index, "Monitor"] == 1:
                self._log(keyword_value)
        self.current_generation += 1    
        return self.current_generation, pd.DataFrame(current_generation)
    
    def _revive_competitiveness(self, competitors: int) -> float:
        return np.random.uniform(1 - self.volatility, 1 + self.volatility) * competitors

    def _revive_volatile_value(self, valatile_value: float) -> float:
        return np.random.uniform(1 - self.volatility, 1 + self.volatility) * valatile_value

    def _randomize_dificulty_score(self, competitiveness: float) -> float:
         return np.clip(competitiveness + np.random.normal(0, 0.05),0,1)     

    def _calculate_ctr(self, total_clicks: int, conversion_clicks) -> float:
        if total_clicks == 0:
            return 0
        return 100 * conversion_clicks / total_clicks                                              

    def batch_build(self, num_samples: int) -> pd.DataFrame:
        all_generations = []
        for i in range(num_samples):
            _, self.last_created_generation = self.build_generation()
            all_generations.extend(self.last_created_generation.to_dict('records'))
        return self.last_created_generation, pd.DataFrame(all_generations)

    def _log(self, keyword_value: KeywordValue):
        print(f'{keyword_value.generation}. Generation von "{keyword_value.keyword}" hat Organic {keyword_value.organic_rank}/{keyword_value.organic_clicks}/{keyword_value.organic_click_through_rate} und paid CHF {keyword_value.ad_spend}/{keyword_value.paid_clicks}/{keyword_value.paid_click_through_rate}')

    def register_site_visit(self, site_visit: SiteVisit):
        keyword_index = self.keywords[self.keywords["Keyword"] == site_visit.keyword].index[0]
        if keyword_index > -1:
            self.keywords.loc[keyword_index, "Clicks"] += 1
            self.keywords.loc[keyword_index, "Revenue"] += site_visit.revenue
            if site_visit.revenue > 0.0:
                self.keywords.loc[keyword_index, "Conversions"] += 1
            if site_visit.referer == "Search":
                self.keywords.loc[keyword_index, "SearchClicks"] += 1
                if site_visit.revenue > 0.0:
                    self.keywords.loc[keyword_index, "SearchConversions"] += 1
            if site_visit.referer == "AdWords":
                # fixme: hier (oder beim generieren) müsste noch geprüft werden, ob überhaupt dieses AdWord in der letzten generation geschaltet wurde
                self.keywords.loc[keyword_index, "AdClicks"] += 1
                self.keywords.loc[keyword_index, "AdRevenue"] += site_visit.revenue
                if site_visit.revenue > 0.0:
                    self.keywords.loc[keyword_index, "AdConversions"] += 1
                #current_add_price = self.last_created_generation.loc["Keyword" == site_visit.keyword, "Ad"]   
                current_add_price = 0.20 # fixme Pricing mdynamisch 0.40 - 1.75?
                self.keywords.loc[keyword_index, "AdSpent"] += current_add_price
            if self.keywords.loc[keyword_index, "Monitor"] == 1:
                print(self.keywords.loc[keyword_index])  


    def get_most_likely_search_terms(self):
        """""
        Liefert die Keywords mit den besten organischen Suchergebnissen zurück. 
        """
        return self.keywords[self.keywords["Competitors"] < 10].sort_values(by="Competitors", ascending=True)["Keyword"]

    def get_current_terms_with_ad_contingent(self):
        """""
        Liefert alle Suchbegriffe zurück, welche noch ein Ad-Contingent haben.
        """
        return self.keywords[self.keywords["AdContingent"] > 0]["Keyword"]


class SiteAnalyticsSimulation:
    def __init__(self, target: KeywordValueOfferSimulation, keywords: pd.Series):
        self.target = target
        self.keywords = keywords
        self.started = False
        # todo Ausbauvariante Landingpages einlesen und über diese Keywords und Kurspreis ermitteln?
    
    def __del__(self):
        self.stop()

    def start(self):
        self.started = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        
    def stop(self):
        self.started = False
        self.thread.join()

    def run(self):
        while self.started:
            origin = np.random.choice(["Search", "AdWords", None], p=[1/3, 1/3, 1/3])
            if origin == "AdWords":
                keywords = self.target.get_current_terms_with_ad_contingent()       
                if len(keywords) > 0:         
                    keyword=keywords.sample().values[0]
                else:
                    origin = None
            if origin == "Search":
                keywords = self.target.get_most_likely_search_terms()
                weights = np.linspace(1, 0.1, len(keywords))  # Höhere Gewichtung für frühere Einträge
                weights /= weights.sum() 
                keyword = keywords.sample(weights=weights).values[0] 
            else:
                keyword=self.keywords.sample().values[0]
            site_visit = SiteVisit(
                url="https://www.bfh.ch/de/weiterbildung/cas/artificial-intelligence/",
                referer=origin,
                # todo: Bei den Keywords müssten noch die Reihenfolge berücksichtgt werden.
                keyword=keyword,
                revenue=np.random.choice([6500.00, 0.00], p=[0.001, 0.999]),
            )
            print(site_visit)
            self.target.register_site_visit(site_visit)
            user_activity_level = np.random.uniform(0.0001, 0.1)
            time.sleep(user_activity_level)
    

# _____[ Main (Test/Simulation des Envs)]___________________________________________________________
if __name__ == '__main__':
    log.start_logger("DigitalMarketingEnv", "dev")
    log.app_logger().info("Starte Simulation des DigitalMarketingEnv")
    print("Simuliere Environment-Dynamic eine Minute lang")
    env = DigitalMarketingEnv()
    time.sleep(60)
    del env
    print("Simulation beendet")



   # Keyword-Erstellung
    #keywords = [f"Keyword_{i}" for i in range(num_samples)]
    # Wettbewerbsfähigkeit (je höher, desto schwerer)
    #competitiveness = np.random.beta(2, 5, num_samples)  # Meist niedrige Werte
    # Schwierigkeitsgrad - ähnlich wie competitiveness
    #difficulty_score = competitiveness + np.random.normal(0, 0.05, num_samples)
    #difficulty_score = np.clip(difficulty_score, 0, 1)  # Werte zwischen 0 und 1
    # Organisches Ranking (1 ist das beste)
    #organic_rank = np.random.randint(1, 11, num_samples)
    # Organische Klicks (mehr Klicks bei besserem Rank)
    #organic_clicks = np.maximum(0, np.random.poisson(5000 / organic_rank, num_samples))
    # Organische click_through_rate (höherer Rank → höhere click_through_rate)
    #organic_click_through_rate = np.clip(np.random.beta(2, 8, num_samples) * (1 / organic_rank), 0.01, 0.3)
    # Paid Clicks hängen von Ad Spend ab
    #ad_spend = np.random.lognormal(mean=7, sigma=1.2, size=num_samples)  # Log-Normal für realistischere Verteilung
    #ad_spend = np.clip(ad_spend, 10, 10000)  # Begrenzung
    #paid_clicks = np.maximum(0, np.random.poisson(ad_spend / 10, num_samples))  # Mehr Budget → mehr Klicks
    #paid_click_through_rate = np.clip(np.random.beta(2, 6, num_samples), 0.01, 0.25)  # click_through_rate zwischen 1% und 25%
    # Kosten pro Klick (CPC steigt mit Competitiveness)
    #CPC = Ad Rank of the ad below yours / your Quality Score + $0.01
    #cost_per_click = np.clip(np.random.normal(competitiveness * 10, 1), 0.1, 10)
    # Konversionsrate (Conversion Rate hängt von click_through_rate ab)
    #conversion_rate = np.clip(np.random.beta(2, 8, num_samples) * (organic_click_through_rate + paid_click_through_rate), 0.01, 0.3)
    # Ad Conversions (abhängig von Paid Clicks und Conversion Rate)
    #ad_conversions = (paid_clicks * conversion_rate).astype(int)
    # Return on Ad Spend (ROAS = Conversion Value / Ad Spend)
    #conversion_value = np.random.lognormal(mean=7, sigma=1.5, size=num_samples)  # Einnahmen sind log-normal verteilt
    #ad_roas = np.clip(conversion_value / np.maximum(ad_spend, 1), 0.5, 5)
    # Cost per Acquisition (CPA = Ad Spend / Conversions)
    # CPA = Total Ads spend / Conversions
    #cost_per_acquisition = np.where(ad_conversions > 0, ad_spend / ad_conversions, np.nan)
    # Impression Share (wie oft erscheinen die Ads)
    #impression_share = np.clip(np.random.beta(5, 2, num_samples), 0.1, 1.0)
    # Empfehlung basierend auf Cost-Per-Acquisition und ROAS
    # ROAS = Conversion value / Cost
    #previous_recommendation = (ad_roas > 1.5) & (cost_per_acquisition < 200)
    #previous_recommendation = previous_recommendation.astype(int)


