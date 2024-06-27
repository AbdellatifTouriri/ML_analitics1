


from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import DatasetForm
# ml_predictor/views.py

from django.shortcuts import render, get_object_or_404
from ml_predictor.models import Dataset
import pandas as pd
# views.py

from django.shortcuts import render
from .models import Dataset
import pandas as pd

# views.py

from django.shortcuts import render
from .models import Dataset
import pandas as pd
# views.py
from django.shortcuts import render
from .models import Dataset
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

# views.py
# views.py
from django.shortcuts import render
from .models import Dataset
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
# views.py
from django.shortcuts import render
from .models import Dataset
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from .forms import GraphForm
from .models import Dataset

from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from .forms import GraphForm
from .models import Dataset

from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from .forms import GraphForm
from .models import Dataset
from django.shortcuts import render
from .models import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import json














from django.shortcuts import render, get_object_or_404

from .models import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import json

# views.py

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .models import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# ml_predictor/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from .forms import DatasetForm,   GraphForm

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import plot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import json
import io
import base64
# ml_predictor/views.py

from .models import Dataset, MLModel





# views.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from django.shortcuts import render, redirect, get_object_or_404
from .models import Dataset  # Import your Dataset model
from .forms import PreprocessingForm  # Import your PreprocessingForm

def data_preprocessing(request, dataset_id):
    dataset = get_object_or_404(Dataset, pk=dataset_id)
    df = pd.read_csv(dataset.file.path)
    if request.method == 'POST':
        form = PreprocessingForm(request.POST)
        if form.is_valid():
            selected_options = form.cleaned_data['preprocessing_options']
            print(selected_options)

            if 'drop_missing' in selected_options:
                df.dropna(inplace=True)


            if 'fillna_mean' in selected_options:
                df.fillna(df.mean(), inplace=True)
            if 'fillna_median' in selected_options:
                df.fillna(df.median(), inplace=True)
            if 'standardize' in selected_options:
                scaler = StandardScaler()
                df[df.columns] = scaler.fit_transform(df[df.columns])
            if 'normalize' in selected_options:
                scaler = MinMaxScaler()
                df[df.columns] = scaler.fit_transform(df[df.columns])
            new_file_path = f'{dataset.file.path}'  # Example: Append '_cleaned' to the original file name
            df.to_csv(new_file_path, index=False)

            # Check if any preprocessing options were selected and applied
         #   if any(option in selected_options for option in ['drop_missing', 'fillna_mean', 'fillna_median', 'standardize', 'normalize']):
                # Redirect to dashboard or any other view upon successful preprocessing
              #  return redirect('dashboard')

    else:
        form = PreprocessingForm()

    return render(request, 'ml_predictor/data_preprocessing.html', {'form': form})









# ml_predictor/views.py

from django.shortcuts import render, get_object_or_404, redirect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import json
from .models import Dataset
from django.shortcuts import redirect, render
from ml_predictor.models import Dataset



# views.py
# views.py

from django.shortcuts import render, redirect
from django.http import JsonResponse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from django.shortcuts import render, redirect, get_object_or_404
from .models import Dataset
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import json


def dataset_graph(request, dataset_id):
    dataset = Dataset.objects.get(pk=dataset_id)
    df = pd.read_csv(dataset.file.path)

    if request.method == 'POST':
        form = GraphForm(request.POST, dataset_path=dataset.file.path)
        if form.is_valid():
            graph_type = form.cleaned_data['graph_type']
            column_x = form.cleaned_data['column_x']
            column_y = form.cleaned_data.get('column_y')

            fig, ax = plt.subplots()

            try:
                if graph_type == 'line':
                    if column_y:
                        df.plot(kind='line', x=column_x, y=column_y, ax=ax)
                    else:
                        return render(request, 'ml_predictor/graphs.html', {
                            'form': form,
                            'error_message': "Y-axis column is required for a line graph."
                        })
                elif graph_type == 'bar':
                    if column_y:
                        df.plot(kind='bar', x=column_x, y=column_y, ax=ax)
                    else:
                        return render(request, 'ml_predictor/graphs.html', {
                            'form': form,
                            'error_message': "Y-axis column is required for a bar graph."
                        })
                elif graph_type == 'scatter':
                    if column_y:
                        df.plot(kind='scatter', x=column_x, y=column_y, ax=ax)
                    else:
                        return render(request, 'ml_predictor/graphs.html', {
                            'form': form,
                            'error_message': "Y-axis column is required for a scatter plot."
                        })
                elif graph_type == 'histogram':
                    df[column_x].plot(kind='hist', ax=ax)
                elif graph_type == 'box':
                    df[[column_x]].plot(kind='box', ax=ax)
                elif graph_type == 'pie':
                    df[column_x].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                graph = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()

                return render(request, 'ml_predictor/graphs.html', {
                    'form': form,
                    'graph': graph
                })
            except Exception as e:
                return render(request, 'ml_predictor/graphs.html', {
                    'form': form,
                    'error_message': f"An error occurred while generating the graph: {str(e)}"
                })
    else:
        form = GraphForm(dataset_path=dataset.file.path)

    return render(request, 'ml_predictor/graphs.html', {
        'form': form
    })



















def dataset_detail(request, dataset_id):
    dataset = Dataset.objects.get(pk=dataset_id)

    # Load the dataset using Pandas (example: CSV)
    df = pd.read_csv(dataset.file.path)

    # Exclude non-numeric columns (assuming patient identifiers are in 'Patient' columns)
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_df = df[numeric_columns]

    # Calculate simple statistics
    stats = {
        'Nombre_de_lignes': len(df),
        'Nombre_de_colonnes': len(df.columns),
        'Moyenne': numeric_df.mean().to_dict(),
        'Ecart_type': numeric_df.std().to_dict(),
        'Minimum': numeric_df.min().to_dict(),
        'Maximum': numeric_df.max().to_dict(),
    }

    # Create a bar chart for mean values
    mean_values = numeric_df.mean()
    data = [
        go.Bar(
            x=mean_values.index,
            y=mean_values.values,
        )
    ]
    graph_div = plot(data, output_type='div', include_plotlyjs=False)

    context = {
        'dataset': dataset,
        'stats': stats,
        'graph_div': graph_div,
    }
    return render(request, 'ml_predictor/dataset_detail.html', context)


def dashboard1(request):
    datasets = Dataset.objects.all()
    dataset_stats = []

    for dataset in datasets:
        # Charger le dataset avec Pandas (exemple : CSV)
        df = pd.read_csv(dataset.file.path)

        # Calculer des statistiques simples
        stats = {
            'nb_lignes': len(df),
            'nb_colonnes': len(df.columns),
            # Ajoutez d'autres statistiques nécessaires
        }

        dataset_stats.append((dataset, stats))

    context = {
        'dataset_stats': dataset_stats,
    }
    return render(request, 'ml_predictor/dashboard.html', context)

def dashboard(request):
    datasets = Dataset.objects.all()
    dataset_stats = []

    for dataset in datasets:
        try:
            df = pd.read_csv(dataset.file.path)

            # Calculate statistics
            stats = {
                'dataset_name': dataset.name,
                'nb_rows': len(df),
                'nb_columns': len(df.columns),
                # Add more statistics as needed
            }

            # Append the dataset object and stats dictionary as a tuple to dataset_stats
            dataset_stats.append((dataset, stats))

        except pd.errors.EmptyDataError:
            # Handle empty file scenarios if needed
            # For example, skip this dataset or log the error
            pass

        except Exception as e:
            # Handle other exceptions gracefully
            # Log the exception for debugging
            print(f"Error processing dataset {dataset.name}: {str(e)}")

    context = {
        'dataset_stats': dataset_stats,
    }

    return render(request, 'ml_predictor/dashboard.html', context)


def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save()
            # Perform data processing steps here
            # Example: Calculate statistics, preprocess data

            # Redirect to a dashboard or result page
            return redirect('dashboard')
    else:
        form = DatasetForm()
    return render(request, 'ml_predictor/upload_dataset.html', {'form': form})
# views.py

from django.shortcuts import get_object_or_404, redirect
from .models import Dataset

def delete_dataset(request, dataset_id):
    dataset = get_object_or_404(Dataset, pk=dataset_id)
    dataset.delete()
    return redirect('dashboard')  # Replace 'dashboard' with the name of your dashboard URL pattern

"""

def dashboard(request):
    datasets = Dataset.objects.all()
    return render(request, 'ml_predictor/dashboard.html', {'datasets': datasets})
"""