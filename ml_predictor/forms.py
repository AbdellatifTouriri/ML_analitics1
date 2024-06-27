from django import forms
from .models import Dataset
import pandas as pd
class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ('name', 'file',)

from django import forms
import pandas as pd

class GraphForm(forms.Form):
    graph_type_choices = [
        ('line', 'Line'),
        ('bar', 'Bar'),
        ('scatter', 'Scatter'),
    ]

    graph_type = forms.ChoiceField(choices=graph_type_choices, label='Type of Graph')
    column_x = forms.ChoiceField(label='X Axis Column')
    column_y = forms.ChoiceField(label='Y Axis Column')

    def __init__(self, *args, **kwargs):
        dataset_path = kwargs.pop('dataset_path', None)
        super().__init__(*args, **kwargs)
        if dataset_path:
            df = pd.read_csv(dataset_path)
            numeric_columns = df.select_dtypes(include=['number']).columns
            column_choices = [(col, col) for col in numeric_columns]
            self.fields['column_x'].choices = column_choices
            self.fields['column_y'].choices = column_choices


# forms.py
# ml_predictor/forms.py

from django import forms

class PreprocessingForm(forms.Form):
    OPTIONS = [
        ('drop_missing', 'Drop Missing Values'),
        ('fillna_mean', 'Fill NaN with Mean'),
        ('fillna_median', 'Fill NaN with Median'),
        ('standardize', 'Standardize Data'),
        ('normalize', 'Normalize Data'),
        # Add more preprocessing options as needed
    ]

    preprocessing_options = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        choices=OPTIONS,
        required=False
    )

