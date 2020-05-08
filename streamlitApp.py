import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import plotly.graph_objects as go
from datetime import date
from datetime import timedelta
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def criar_curvas_norm(df_dados, codigo_acao):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_dados.index, y=df_dados.AcaoNorm, mode='lines', name=codigo_acao.upper()))
    fig.add_trace(go.Scatter(x=df_dados.index, y=df_dados.IbovNorm, mode='lines', name='IBOV'))
    return fig

def criar_dispersao(df_dados,codigo_acao):
    # ======================================================
    fig = go.Figure(data=go.Scatter(x=df_dados[df_dados.columns[3]], y=df_dados[df_dados.columns[9]], mode='markers'))

    fig.update_layout(
        xaxis_title='Fechamento ' + str(codigo_acao),
        yaxis_title="Fechamento IBOV",
    )
    return fig
    # ======================================================

def criar_candlestick(df_dados,ver_acao,codigo_acao):
    # ======================================================
    if ver_acao:
        alta = df_dados[df_dados.columns[0]]
        baixa = df_dados[df_dados.columns[1]]
        fechamento = df_dados[df_dados.columns[3]]
        abertuta = df_dados[df_dados.columns[2]]
        titulo = codigo_acao
    else:
        alta = df_dados[df_dados.columns[6]]
        baixa = df_dados[df_dados.columns[7]]
        fechamento = df_dados[df_dados.columns[9]]
        abertuta = df_dados[df_dados.columns[8]]
        titulo = 'IBOV'

    fig = go.Figure(data=[go.Candlestick(x=df_dados.index, open=abertuta, high=alta, low=baixa, close=fechamento)])

    fig.update_layout(
             title=titulo,
        shapes=[dict(
            x0='2020-03-01', x1='2020-03-01', y0=0, y1=1, xref='x', yref='paper',
            line_width=2)],
        annotations=[dict(
            x='2020-03-01', y=0.05, xref='x', yref='paper',
            showarrow=False, xanchor='left', text='Covid19')]
    )

    return fig

 # ======================================================
def previstosxhistoricos(data):
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    fig = go.Figure()
    # fig.add_trace(go.Scatter( y=train['Close'],mode='lines+markers',name='Treino'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines+markers', name='Valores Históricos'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines+markers', name='Previsões'))

    fig.update_layout(
        title={
            'text': "Valores Previstos e Valores Históricos",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    return fig


# ======================================================

def realizarPrevisao(df,codigo_acao):
    st.header("Histórico")
    titulo = 'Histórico Preço de Fechamento - ' + str(codigo_acao).upper()
    plt.figure(figsize=(16, 8))
    plt.title(titulo)
    plt.plot(df['Close'])
    #plt.xticks(rotation='vertical')
    # plt.xticks(df.index)
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Preço de Fechamento BRL (R$)', fontsize=18)
    st.pyplot()

def main():
    sucesso=False
    sucesso2 = False
    codigo_acao=""
    st.title('Análise de Ações')
    st.subheader('Objetivos')
    st.text('Analisar os dados de uma ação e correlacionar seu comportamento com o índice Ibovespa.')
    st.text('Prever o preço de fechamento de uma ação baseado em seus últimos 60 dias')
    st.image('https://media.giphy.com/media/l0HlDDyxBfSaPpU88/giphy.gif', width=700)

    st.sidebar.title('AceleraDev Data Science')
    st.sidebar.image('logo.png', width=280)
    st.sidebar.subheader("By Pável Lelis")
    st.sidebar.title("O que fazer?")
    app_mode = st.sidebar.selectbox("",["","Análise Exploratória","Prever o Fechamento","Ver o Workflow deste App"])

    if app_mode == "Análise Exploratória":
        #st.sidebar.success('Para continuar digite o código da ação.')
        codigo_acao = st.sidebar.text_input('Digite o código da ação aqui')
        datainicio=st.sidebar.date_input('Data de início')
        datafim = st.sidebar.date_input('Data Final')
        #botao=st.sidebar.button("Carregar Dados")
        #if botao:

        # Requisição e carga de Dados
        if codigo_acao!="" and datainicio!=datafim:
            codigo_acao2 = codigo_acao + ".SA"
            df = web.DataReader(codigo_acao2, data_source='yahoo', start=datainicio, end=datafim)
            df_ibovespa = web.DataReader('^BVSP', data_source='yahoo', start=datainicio, end=datafim)
            df_quote = df
            sucesso=True
            # Padronização do DataFrame
            for c in df_ibovespa.columns:
                df_ibovespa.rename(columns={c: c + '_IBOV'}, inplace=True)
            for c in df_quote.columns:
                df_quote.rename(columns={c: c + '_' + codigo_acao.upper()}, inplace=True)
            df_dados = pd.merge(df_quote, df_ibovespa, left_index=True, right_index=True)

            def normalizaIbov(x):
                return (x - (df_dados[df_dados.columns[9]]).min()) / (
                        (df_dados[df_dados.columns[9]]).max() - (df_dados[df_dados.columns[9]]).min())

            def normalizaAcao(x):
                return (x - (df_dados[df_dados.columns[3]]).min()) / (
                        (df_dados[df_dados.columns[3]]).max() - (df_dados[df_dados.columns[3]]).min())

            df_dados['AcaoNorm'] = df_dados[df_dados.columns[3]].apply(normalizaAcao)
            df_dados['IbovNorm'] = df_dados[df_dados.columns[9]].apply(normalizaIbov)

    elif app_mode == "Prever o Fechamento":
        if codigo_acao=='':
            codigo_acao = st.sidebar.text_input('Digite o código da ação aqui')
            sucesso2=True
        else:
            escolhaPrevisao = st.sidebar.selectbox("Qual fechamento vamos prever?", [codigo_acao, 'Ibov',"Outra Ação"])
            if escolhaPrevisao == "Outra Ação":
                codigo_acao = st.sidebar.text_input('Digite o código da ação aqui')
                sucesso2 = True
            elif escolhaPrevisao=='Ibov':
                codigo_acao='^BVSP'
                sucesso2 = True
            else:
                pass




    elif app_mode == "Ver o Workflow deste App":
        st.image("App Workflow.png",width=850)

    if sucesso:
        st.title(codigo_acao)
        st.text('Os dados do IBOV e da ação de análise estão apresentados pela data mais recente.')
        num_linha = st.slider('Registros', 5, len(df_dados))
        st.dataframe(df_dados.tail(num_linha))
        st.title("Gráficos")
        selecionaGrafico=st.selectbox('Escolha o gráfico para análise',['','Candlestick','Dispersão IBOV x Ação','Curvas Normalizadas',])
        if selecionaGrafico == "Candlestick":
            papel=st.selectbox('Para qual histótico?',['',codigo_acao,'IBOV'])
            if papel!="":
                ver_acao=False
                if papel==codigo_acao:
                    ver_acao=True
                st.header("CandleStick")
                st.write(criar_candlestick(df_dados,ver_acao,codigo_acao))
        if selecionaGrafico == "Dispersão IBOV x Ação":
            st.header("Dispersão IBOV x Ação")
            st.write(criar_dispersao(df_dados, codigo_acao))
        if selecionaGrafico == "Curvas Normalizadas":
            st.header("Curvas Normalizadas")
            st.write(criar_curvas_norm(df_dados, codigo_acao))

    if sucesso2:
        # Requisição e carga de Dados
        codigo_acao2 = codigo_acao + ".SA"
        hoje = date.today()
        intervalo = timedelta(1200)
        passado = hoje - intervalo
        if codigo_acao!="":
            df = web.DataReader(codigo_acao2, data_source='yahoo', start=passado, end=hoje)

            texto = 'Realizando a previsão do próximo fechamento de '+ codigo_acao
            st.title(texto)
            realizarPrevisao(df,codigo_acao)
            ##################################################
            st.text("Processando dados...")
            # Criando um novo dataframe com a coluna Fechamento 'Close'
            data = df.filter(['Close'])
            # Criando um vertor de dados
            dataset = data.values
            # obtendo o número de registros para o conjunto de treino (Premissa 80/20)
            training_data_len = math.ceil(len(dataset) * .8)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            # Criando o dataset normalizado
            train_data = scaled_data[0:training_data_len, :]
            # Separando datasets x_train e y_train
            x_train = []
            y_train = []
            for i in range(60, len(train_data)):
                x_train.append(train_data[i - 60:i, 0])
                y_train.append(train_data[i, 0])

            # Criando numpy Arrays
            x_train, y_train = np.array(x_train), np.array(y_train)

            # Formatando os dados de acordo com o formato aceitado pelo LSTM
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            st.text("Configurando o Modelo...")
            # Configurando o Modelo LSTM

            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dense(units=25))
            model.add(Dense(units=1))
            st.text("Compilando o Modelo...")
            # Compilando o modelo
            model.compile(optimizer='adam', loss='mean_squared_error')
            st.text("Treinando o modelo...")
            # Treinando o modelo
            model.fit(x_train, y_train, batch_size=1, epochs=1)
            # Verificando o ajuste do modelo
            st.text("Verificando o ajuste do modelo...")
            test_data = scaled_data[training_data_len - 60:, :]
            # Criando x_test e y_test
            x_test = []
            y_test = dataset[training_data_len:, :]
            for i in range(60, len(test_data)):
                x_test.append(test_data[i - 60:i, 0])
            # Convertendo x_test a numpy array
            x_test = np.array(x_test)

            # Formatando os dados de acordo com o formato aceitado pelo LSTM
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)  # Desfazendo a normalização

            # Plot/Create the data for the graph
            train = data[:training_data_len]
            valid = data[training_data_len:]
            valid['Predictions'] = predictions
            # Visualize the data
            fig=plt.figure()
            plt.figure(figsize=(16, 8))
            plt.title('Modelo', fontsize=36)
            plt.xlabel('Data', fontsize=18)
            plt.ylabel('Preço de Fechamento (BRL)', fontsize=18)
            plt.plot(train['Close'])
            plt.plot(valid[['Close', 'Predictions']])
            plt.legend(['Treino', 'Valores', 'Previsões'], loc='lower right')
            st.pyplot()
            ##################################################
            def previstosxhistoricos(data):
                valid = data[training_data_len:]
                valid['Predictions'] = predictions
                fig = go.Figure()
                # fig.add_trace(go.Scatter( y=train['Close'],mode='lines+markers',name='Treino'))
                fig.add_trace(
                    go.Scatter(x=valid.index, y=valid['Close'], mode='lines+markers', name='Valores Históricos'))
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines+markers', name='Previsões'))

                fig.update_layout(
                    title={
                        'text': "Valores Previstos e Valores Históricos",
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})

                return fig

            st.write((previstosxhistoricos(data)))

            st.text("Realizando a previsão do próximo fechamento...")
            #############################################################
            # Get the quote
            codigo_acao = codigo_acao  # +".SA"
            quote = web.DataReader(codigo_acao + ".SA", data_source='yahoo', start=passado, end=hoje)
            # Create a new dataframe
            new_df = quote.filter(['Close'])
            # Get teh last 60 day closing price
            last_60_days = new_df[-60:].values
            scaler = MinMaxScaler(feature_range=(0, 1))

            # Scale the data to be values between 0 and 1
            last_60_days_scaled = scaler.fit_transform(last_60_days)
            # Create an empty list
            X_test = []
            # Append teh past 60 days
            X_test.append(last_60_days_scaled)
            # Convert the X_test data set to a numpy array
            X_test = np.array(X_test)
            # Reshape the data
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            # Get the predicted scaled price
            pred_price = model.predict(X_test)
            # undo the scaling
            pred_price = scaler.inverse_transform(pred_price)
            if (round((pred_price.tolist()[0][0]), 2)) > valid.Close.tail(1)[0]:
                resp = 'Vai Subir'
            else:
                resp = 'Vai descer'

            valor_previsao= "O valor predito para o próximo fechamento é de "+ str(round((pred_price.tolist()[0][0]), 2)) + ". " + resp+ "! "
            st.title(valor_previsao)
            st.image('Nota.png',width=600)




if __name__ == '__main__':
    main()