import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pickle
#import utils.custom_functions as cf
from sklearn.base import clone
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn", layout="wide", initial_sidebar_state="expanded", menu_items=None)
#st.set_option('deprecation.showPyplotGlobalUse', False)
config = {'displayModeBar': False, "showTips": False    }

PATH = "data/df_plots/"
def load_df(file_name):
    df = pd.read_csv(PATH+file_name)
    return df


html = """<p>
    <p>
    <p>
    <p>
      <a href="https://www.linkedin.com/in/vinicius-torres-05a35695/" rel="nofollow noreferrer">
        <img src= "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAABHtJREFUWEelV09olEcU/80mWEuhQmg9xOKhoSBW46EgPUaR4kUPbVOPPUqhhx68GdHtBgr14slDDx4CZl1LkZpDL202NQUPRWhNtnV3I4n5Z3ZNNFRSGt3syJuZN9/MfPNtAp3Lfvt9b9783u/93psZARoXZQ7d4wuA6FX/JQChnoJBL+lj5L2UwZzENj1LLqF1fD/yoi0weLML/W+1rMvMxcnCuOpoYzyxTZbt6gKw/Hq3QOGXJRs5o/AmZUUdY0i/c2f4s80/WnxtERCiKVAY9zglky+PvoOT7/Zg4tE6Lt+dT5wqSwEIdwrnyk9NJmxe3OB3AOgpcmggJQAxXPYjy6B1O67E6gIkRc6DYmEGaPKnB/fi5sfvp7j9/Me/MTK1krznYKNCNSyFoqTIny4qDTNQ+vVScO7D/bh8oi8F4Ovf5nCxPEs5y6gCf4q2cgS7pnMemy1EYZywKmS5HLB1/pjvTQK7vpnAy3as/LKFaL8EOU8XcCDCg2+/gcrZo9au/7vfMdXcSGorkv84LwJYnddqD0vDQeFpwNe2gEw1l52VnsqyWbxz0ihRAQM7a0gcksmdI2tBlRSl3VAXIEoBkEO+BmhabrhsBUTflSsnFflfZ3Fpcg5XPnoPJ/dsoF6vK0hs8u3deUzOr0fpSwAYawtA0a/rjPuAcsoAHHf5O7P44oN9eLY4axcPV6uu/YtzP8+4LcAA5BS4AAKhiULZbjTtoWNe9OTx+WYLy3MP9eLsR1J+Vaexg7rqHWbCpEKgUDbak2qyvGAodpVqOqFmwHRKB2S1Ws2M3GVCQuJ0aSqpKAUxEGGoAZ2CCduAwu+8+NP/XoI6Jo3unMCtwcNeFhjv6dJ9b0PfIQC9F7AG+NmN/FTpfrKgAHZ35fD9J4d8KUiJ/OQc7j1+bkH4AEwKXAXTM1WBfifQHhpQaaXFa/W6ev5pZg1X7y2lVD52pj91urn2x2Pcqj5xxFgYl25pxlRuU6AADqBaq6Fe06VGY3S6gWKlker1GoA/rk+v4Eal6WxIVgO8HTsi5N7hiPDBYK+NnJkqGgAmSXY77wSAbeMa4LowVWT7wOoCxk70pKLyGHDo3B5AsB2HIlMHFAitAdNeyakmJqlDBkDz7V4vgNufZaeAo9iRCMVXI3ZXi0U1OtVA8a+GIyz9eJs0wH1bCpDYitMrSi8MNbMMOT5S+4HhH6zzKACrAf+olLKVwGhFizCxjDYikrtAraZLza3xOIAmihXnyGbyENrSoqOmCuIpAEydC1XnvKtpAHpzGnPyyixdn27ghqLVHRJjZ45Ey7C0HQPVag31es1OPlX6s2NpKRGSBoJTmwIbHFy5D2Qy0LPRQKv5yEEu8M+LllI9VcSbu7rNc2KyuSWxudVOMUC2ajilGdoaERqLJ+boHLgyLcq/L3Q4a4WfOh3L6GpGd6R9XOcdbqapfP7/F6Ih1M14fWTLnl4jXlUE7m3CyXXG60xsHht9L7q0ROiG/NrMMiD3atHoMoyNxEHG/Ww7WnQ0DfRt9iKfb78CLOl4lgOx5t8AAAAASUVORK5CYII="> LinkedIn
      </a> &nbsp; 
      <a href="https://github.com/vinitg96/Projetos_Data_Science/tree/main/Analise_Rotatividade_Clientes_Telco" rel="nofollow noreferrer">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAABMpJREFUWEeVV02IHEUUfq97Z8yCIsp6iGgCih7iHhREAzNVs+shB8lBECXJKQchEZUkm+lkL7IQxOD07GYvrhoPIvgvKOrFHKI71bPgQQWJwYMREveirAEVnSTsTD2prv6p7q7uWesyM/WqXr33ve99VYNQNTAyUrqoOR/cRpt40HXkDADu0Ba8QlL23NHordXl2T/DGQAwtmVOSWyo140dzaNiO9TwGwdgBwIBUXZn8TC6XKvVd59/effvpnNzXfw9DUD5pGzUjZP9W1wpNwDwpnyUVRkaa687g+Edqyuz/9jhwBwCkVf1wbzVZQD3yDg4x8KnSoGwFHTYce0+G3qEQHay5YlLBHivPUtV2S1VziAQAhBdEl1+X95nzhMC93rrSHiXKnM6YjoZk0YcFg6UBfmr8NnO1K9RAvWVe70vAHCvXkARXPFyfWLVYVvhBQF+FvjNJ2KvSUoz7bVpifKCmbcEerrv848bJ/t3OpIuIMDtmqZmf+rvucpedaA+3fMf+a3R7h9wUL5rlm00xOm15eZF5SoJoOUFlO9b4bNMIdic2IUOHgAp3xBLrXUMS6tD5nO9u8nFww7SO71O66ckEQTgbWEeFZoCn6EiZHgAmxPPoosreUYnAZhNGwNg0qJMcSKHvB2E5M+gJOGQWGRnwwC4FxRcKE3odRoOOE5oi6sfknPMgZlEFsjhg/4omTMqqFDAmYWvt8nBxDVbv9cm6lPnTz961YSzTFAyURkUmTn+7ZR0rm3Y9MLZGE4i94IuABy3BPCq8NnzacKRjKi66yaxDJOckTnkQPA6ABzKbyBCH7knBgA4mTUSCJ9Xqk0Yzv8ohy5zQcQGCgFrLvkOsOVbfeeZnU7AvWInhKjHDLV2gJlhLlsTnnxj2IItJBptqkTAptv5y8SOTJFRZUiHCISCkvOkSpBQqrTWMTERqCBjWYca6TwHCBQJCyqVKtUWiR4mYMii5UWUQcC8yJgnrjjR0yqHwkvCZy+Oh7iCKNHmphecdoDm89c4AV7GphcccQCWbQdZO8E4z1qZZDJ9YmXrn6ZPRC+USrEKSBJs9rusXo5C1mILiLfFJiBO5H2otb2YZ7wdDAHBDRcR/QGINwPAtmSTlAeDxdbbW7oCoigannjGBXyzLHgCGAY+q2kE5noPoet8ryWWhOjyVnM+uMcZwS8ZB0QjdPHT3ivsqaJj9aARnwPRXkDdQFVCKUfwYH+J/5DoCW+L64D69UsEf/e77NaG4gfBcuIJARyC2dUuW7WLzdoeAHkutcXqm1N1ohuiy0OEE8uuhY/qU4PtN5LNo9GTYmnmE+aJJQQ4Fi8dR8wywTED/vevyfp3Zx/ezASgfrTawX5CeE9Dl15IljsuA4B5Y2pdqRq4T/jND+MVCOa7Kvw/IF5DwMMRIX8WXX5/ZTFzoWQCiEJJlBZhJeiw58wIjeKkebATa6eQZCJCRCABYB2RdhavafVWUnjpoUqgf2frTkCnAp8v2NoxnEv1Q0tq49hXD7gTtR9TqPR9UeBAjuo2DoyGm9NrZx67mPeV5UBJz7ROBO8Twb7wfyMCiC5TCZaOlicozh4RPuh12H5zcYqLiZPt8MwcAmuLo+jAHtFhj1dRjHvBOSnpy/4iP1NNRm39D5AhBpJuROh9AAAAAElFTkSuQmCC" alt="github"> Github
        <i class="fa-brands fa-medium"></i>
      </a>
    </p>"""



    
st.title("Rotatividade de Clientes em Empresas de Telecomunica????o")

sidebar = st.sidebar.radio("Se????es",options=("Dashboard","Insigths e Sugest??es","Cadastro de Novo Cliente"))

st.sidebar.write("O objetivo desse projeto foi avaliar dados reais de rotatividade em empresas de telecomunica????o, buscar insigths atrav??s de visualiza????es e propor solu????es que possam auxiliar os tomadores de decis??o a por em pr??tica medidas afim de reduzir a evas??o de clientes.")
st.sidebar.write("Tamb??m foram empregradas t??cncias de aprendizado de m??quina para obter a probabilidade de um cliente vir a deixar a empresa (aprendizagem supervisionada) e segmentar clientes em grupos com base em seus comportamentos (aprendizagem n??o-supervisionada).")

with st.sidebar:
        st.markdown(html, unsafe_allow_html=True)

if sidebar == "Dashboard":

    col1,col2,col3 = st.columns(3)

    col1.metric("Clientes na base de dados", "5.000")
    col2.metric("Ativos", "4.293 (85,9%)")
    col3.metric("Evas??o","707 (14,1%)")

    ############# Plot dos Mapas

    df_map = load_df("df_map.csv")



    def mapplot(df_map, user_choice, sufix = ""):
        
        fig = px.choropleth(df_map,
        locations='Sigla',
        color= user_choice,
        color_continuous_scale='ice',
        hover_name='Estado',
        hover_data = {user_choice:True,"Sigla":False},
        basemap_visible = True,
        locationmode='USA-states',
        labels={'Evas??o':'Evas??o'},
        scope = "usa"
                )

        fig.update_layout(
        title='',
        margin=dict(l=0, r=0, t=30, b=0),
        height = 400
        )

        fig.update_coloraxes(
        showscale = True,
        autocolorscale = False,
        reversescale = True,
        cmin = df_map[user_choice].min(),
        cmax = df_map[user_choice].max(),
        cmid= df_map[user_choice].median(),
        colorbar = dict(orientation = "h", tickfont_size = 15, thickness = 15, lenmode = "fraction", len = 0.75,
                        ticksuffix = sufix, ypad = 5, xpad = 10, title_side = "top"),
        colorbar_title_text=""
                )
        
        return fig


    col1,col2 = st.columns(2)

    with col1:
        choice = st.selectbox("", ["Arrecada????o", "Chamadas", "Clientes", "Rotatividade"],key = "left", index = 3)

        if (choice == "Arrecada????o") or (choice == "Rotatividade"):
            st.subheader(f"{choice} m??dia por estado")
        else:
            st.subheader(f"M??dia de {choice} por estado")
        
        if choice == "Rotatividade":
            map1 = mapplot(df_map, choice, sufix="%")
        else:
            map1 = mapplot(df_map, choice)

        st.plotly_chart(map1, use_container_width=True)

    with col2:
        choice2 = st.selectbox("", ["Arrecada????o", "Chamadas", "Clientes", "Rotatividade"],key = "right", index = 2)

        if (choice2 == "Arrecada????o") or (choice2 == "Rotatividade"):
            st.subheader(f"{choice2} m??dia por estado")
        else:
            st.subheader(f"M??dia de {choice2} por estado")

        if choice2 == "Rotatividade":
            map2 = mapplot(df_map, choice2, sufix="%")
        else:
            map2 = mapplot(df_map, choice2)

        st.plotly_chart(map2, use_container_width=True)

    ######### Barplot Horario

    minutes_df = load_df("minutes_barplot_df.csv")
    calls_df = load_df("calls_barplot_df.csv")
    arrec_df = load_df("arrec_barplot_df.csv")

    def barplot(df, user_choice):
        fig = px.bar(df, x  = "periodo", y = "values", hover_data = {"periodo":False}, labels = {"values":user_choice, "periodo":""})
        return fig

    container = st.container()
    choice3 = st.selectbox("",["Arrecada????o", "Chamadas", "Minutos"], index=0)

    with container:
        if choice3 == "Arrecada????o":
            st.subheader(f"{choice3} m??dia por per??odo do dia")
        else:
            st.subheader(f"M??dia de {choice3} por per??odo do dia")

    if choice3 == "Arrecada????o":
        bar = barplot(arrec_df, choice3)
    elif choice3 == "Chamadas":
        bar = barplot(calls_df, choice3)
    else:
        bar = barplot(minutes_df, choice3)

    st.plotly_chart(bar, use_container_width=True)


    ############ Pie plot ############
    st.subheader("Percentual dos clientes que deixaram a empresa com rela????o aos planos adquiridos")

    pie_df = load_df("plan_pie.csv")

    pie_plot = px.pie(pie_df, values="n", names="label", hover_data = {"n":True, "label":False})

    pie_plot.update_layout(title='', margin=dict(l=0, r=0, t=30, b=0), font=dict(size=17), height = 450, width=550,
                        legend_title_side="top", legend_x=0.8, hoverlabel = dict(font=dict(size=17)))

    pie_plot.update_traces( textfont_color="#000000")

    st.plotly_chart(pie_plot, use_container_width=True, config = config)

    



    ########### Sonarplot clusters #############

    sonar_df = load_df("sonar_df.csv")

    go.Figure()

    sonar = go.Figure()

    sonar.add_trace(go.Scatterpolar(
    r=sonar_df["0"],
    theta=sonar_df.iloc[:,0],
    fill='toself',
    line = dict(color = "#33FFE3"),
    name='Grupo 1',
    hoverinfo = "none"
    ))
    sonar.add_trace(go.Scatterpolar(
    r=sonar_df["1"],
    theta=sonar_df.iloc[:,0],
    fill='toself',
    name='Grupo 2 ',
    hoverinfo = "none"
    ))

    sonar.add_trace(go.Scatterpolar(
    r=sonar_df["2"],
    theta=sonar_df.iloc[:,0],
    fill='toself',
    name='Grupo 3',
    hoverinfo = "none"
    ))

    sonar.add_trace(go.Scatterpolar(
    r=sonar_df["3"],
    theta=sonar_df.iloc[:,0],
    fill='toself',
    name='Grupo 4',
    hoverinfo = "none"
    ))

    sonar.update_layout(
    polar=dict(
    radialaxis=dict(
    visible=True,
    range=[0, 5]
    )),
    font=dict(size=17),
    showlegend=True, margin=dict(l=0, r=0, t=60, b=30), height = 450, legend_x=0.8
    )
    st.subheader("Segmenta????o dos Clientes em Grupos")
    st.write("Valores redimensionados em uma escala de pontua????o de 0 a 5")
    st.plotly_chart(sonar, use_container_width=True)

if sidebar == "Cadastro de Novo Cliente":

    st.subheader("Preencha com os dados do cliente para obter a probabilidade de evas??o")

    ########### model
    train_fe = pd.read_csv("data/train_fe.csv")
    test_fe = pd.read_csv("data/test_fe.csv")
    data = pd.concat([train_fe, test_fe]).reset_index(drop=True)
    data = data.drop("churn", axis = 1)
    data["international_plan"] = data["international_plan"].map({0:"N??o", 1:"Sim"})
    data["voice_mail_plan"] = data["voice_mail_plan"].map({0:"N??o", 1:"Sim"})
    data["area_code"] = data["area_code"].map({"area_code_408":"408", "area_code_415":"415", "area_code_510":"510"})


    cols = data.columns.to_list() ## modelo foi treinado com variaveis nessa ordem
    derivate_vars = ["rechargesPerMinute","minutes_total","calls_total","recharge_total"]
    to_selectbox = ["international_plan", "voice_mail_plan", "state","area_code"]
    to_slider = list( ((set(cols) - set(to_selectbox))) - set(derivate_vars) )
    to_slider = sorted(to_slider)

    to_selectbox_label = ["Plano Internacional", "Plano de Correio de Voz", "Estado", "C??digo de ??rea"]
    to_selectbox_help = ["Cliente possui ou n??o plano internacional", "Cliente possui ou n??o plano de correio de voz",
    "Estado do Cliente", "C??digo de ??rea do Cliente (3 digitos)"]


    to_slider_label = ["Dura????o da Conta", "Liga????es para Atendimento ao Cliente","Mensagens por Correio de Voz", "Liga????es Diurnas",
    "Gasto Diurno", "Minutos Diurnos", "Liga????es no Per??odo da Tarde", "Gasto no Per??odo da Tarde", "Minutos no Per??odo da Tarde",
    "Liga????es Internacionais", "Gasto Internacional", "Minutos Internacional", "Liga????es Noturnas", "Gasto Noturno", "Minutos Noturnos"]
    to_slider_help = ["N??mero de meses que o cliente est?? com o provedor de telecomunica????es atual",
    "N??mero de liga????es para atendimento ao cliente",
    "N??mero de mensagens por correio de voz",
    "Total de liga????es realizadas no per??odo da manh??",
    "arrecn??a total de chamadas realizadas pela manh??",
    "Total de minutos gastos em chamadas realizadas pela manh??",
    "Total de liga????es realizadas no per??odo da tarde",
    "arrecn??a total de chamadas realizadas pela tarde",
    "Total de minutos gastos em chamadas realizadas pela tarde",
    "Total de liga????es realizadas para fora do pa??s",
    "arrecn??a total de chamadas realizadas para fora do pa??s]",
    "Total de minutos gastos em chamadas realizadas para fora do pa??s",
    "Total de liga????es realizadas no per??odo da noite",
    "arrecn??a total de chamadas realizadas a noite",
    "Total de minutos gastos em chamadas realizadas a noite"
    ]


    def make_selectbox(label, varname, help):
        selectbox = st.selectbox(label, options = np.unique(data[varname]), help = help)
        return selectbox

    def make_slider(label, varname, help):
        min = int(data[varname].min())
        max = int(data[varname].max())
        med = int(data[varname].median())
        slider = st.slider(label,min,max,med, help = help)
        return slider

    dic = {}


    for catlabel, catvar, cathelp in zip( to_selectbox_label, to_selectbox, to_selectbox_help):
        selectbox = make_selectbox(catlabel, catvar, cathelp)
        dic[catvar] = selectbox

    for numlabel, numvar, numhelp in zip(to_slider_label, to_slider, to_slider_help):
        slider = make_slider(numlabel, numvar, numhelp)
        dic[numvar] = slider

    #dic

                    
    order_dic = {}

    for i in (cols[:-4]):
        for v,j in dic.items():
            if i == v:
                order_dic[i] = j

    if order_dic["international_plan"] == "Sim":
            order_dic["international_plan"] = 1
    else:
            order_dic["international_plan"] = 0

    if order_dic["voice_mail_plan"] == "Sim":
            order_dic["voice_mail_plan"] = 1
    else:
            order_dic["voice_mail_plan"] = 0

    if order_dic["area_code"] == "415":
            order_dic["area_code"] = "area_code_415"
    elif order_dic["area_code"] == "408":
            order_dic["area_code"] = "area_code_408"
    else:
            order_dic["area_code"] = "area_code_510"

    #new features features finais
    order_dic["recarga"] = order_dic["total_day_charge"] + order_dic["total_eve_charge"] + order_dic["total_night_charge"] + order_dic["total_intl_charge"]
    order_dic["minutos"] = order_dic["total_day_minutes"] + order_dic["total_eve_minutes"] + order_dic["total_night_minutes"] + order_dic["total_intl_minutes"]
    order_dic["chamadas"] = order_dic["total_day_calls"] + order_dic["total_eve_calls"] + order_dic["total_night_calls"] + order_dic["total_intl_calls"]
    order_dic["rechargesPerMinute"] = order_dic["recarga"] / order_dic["minutos"]


    #order_dic

    new_data = np.array([i for i in order_dic.values()]).reshape((1,-1))
    new_df = pd.DataFrame(new_data, columns=cols)
    #new_df

    model = pickle.load(open("trained_models/XGB.pickle.dat", "rb"))
    

    if st.button("Obter previs??es"):
    


        prob = model.predict_proba(new_df)[0][1]

        to_drop_cluster = ["state","area_code","international_plan","voice_mail_plan"]
        df_cl = new_df.drop(to_drop_cluster, axis = 1)
        df_cl["prob"] = prob

        km = pickle.load(open("trained_models/kmeans.pickle.dat", "rb"))

        cl = km.predict(df_cl)[0]

        cl +=1


        st.write(f"O cliente tem {prob*100:.2f}% de chance de deixar a empresa e pode ser enquadrado no Grupo {cl}")


        ###############interpretability
        # temp = cf.get_feature_names(model[0])
        # cols = []

        
        # for i in range(len(temp)):
        #     cols.append(temp[i][temp[i].find("__")+2:])

        # train_fe = pd.read_csv("data/train_fe.csv")
        # test_fe = pd.read_csv("data/test_fe.csv")
        # data = pd.concat([train_fe, test_fe]).reset_index(drop=True)


        # X = data.drop("churn",axis = 1)
        # y = data["churn"]


        # XGB = clone(model[2])
        # X_tr = pd.DataFrame(model[0].transform(X), columns = cols)
        # XGB.fit(X_tr,y)

        # explainer = shap.Explainer(XGB)
        # new_data_tr = pd.DataFrame(model[0].transform(new_df), columns = cols)
        # shap_values = explainer(new_data_tr)

        # new_data_tr.to_csv("new_data_tr.csv", index = False)


        
        # ax = shap.plots.force(shap_values, out_names = "kapa", feature_names=None, matplotlib = True)
        # st.pyplot(ax)
if sidebar == "Insigths e Sugest??es":
    st.subheader("Insigths obtidos e sugest??es propostas")
    st.markdown("- Washignton, California e Nova Jersey apresentaram mais de 25% de evas??o.  Desses, a California , apesar de ser o estado mais populoso do p??is, foi o local com menor num??ro de clientes (52) al??m de menor arrecada????o dentre todos os estados. Possivelmente, a forte concorrencia enfrentada ?? um fator determinanate para essa alta rotatibilidade. Logo, devido ao baixo retorno e dificuldade de manter novos clientes nessas regi??es, especialmente na Calif??rnia, talvez seja mais interessante direcionar esfor??os a outras locais com maior potencial de retorno.")
    st.markdown("- A Virginia Ocidental ?? o estado com maior num??ro de clientes (158) e possui uma taxa de evas??o razo??vel (14%), entretanto a arrecada????o m??dia foi relativamente baixa quando comparado com outros estados. Florida (13% de evas??o), Dakota do Norte (11% de evas??o) e Idaho (11% de evas??o) s??o os estados que mais consumem o servi??o (maior numero de chamadas) e tamb??m apresentaram uma baixa arrecada????o m??dia. Os dados mostram que o servi??o ?? bem aceito nesses estados abrindo a possibilidade de explorar um pequeno reajuste afim de aumentar a arrecada????o nesses locais.")
    st.markdown("- Estados com menor taxa de rotatividdade como Wyoming, Nebraska, Arizona, Illinois, Virginia est??o associado a uma menor arrecada????o mas n??o necessariamente a um menor numero de clientes e por isso, s??o locais importanmtes para consolidar a marca conquistando uma maior fidelidade do consumidor.")
    st.markdown("- Dos clientes que abandonaram a ompanhia, 60% n??o possuiam nenhum dos planos ofertados (mensagens por correio de voz e internacional) e menos de 10% possuiam os dois planos. Seguindo essa tend??ncia, tornar esses planos mais acess??veis, diminuindo a barreira de entrada, parace ser uma medida interessante para diminuir a rota????o de clientes.") 
    st.markdown("- Com base em seus h??bitos de consumo, os clientes podem ser segmentados em quatro grupos. Chama aten????o a diferen??a do grupo que cont??m os clientes com alta probabilidade de deixar a empresa para os demais. Isso mostra que esses clientes tem um comportamento peculiar que permite diferencia-los.")     



	
