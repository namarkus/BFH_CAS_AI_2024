# Vorhaben: Dart Funktion benutzen, die zeigt, welche Parameter (Target, Covarianten usw.) benutzt werden.
# URL:  https://unit8co.github.io/darts/generated_api/darts.explainability.tft_explainer.html

from darts.datasets import AirPassengersDataset

from darts.explainability.tft_explainer import TFTExplainer

from darts.models import TFTModel

series = AirPassengersDataset().load()

model = TFTModel(

    input_chunk_length=12,

    output_chunk_length=6,

    add_encoders={"cyclic": {"future": ["hour"]}}

)

model.fit(series)

# create the explainer and generate explanations

explainer = TFTExplainer(model)

results = explainer.explain()

# plot the results

explainer.plot_attention(results, plot_type="all")

explainer.plot_variable_selection(results)