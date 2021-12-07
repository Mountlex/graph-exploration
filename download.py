# Download osm maps

import osmnx as ox

def download(place,filename=None):
    if not filename:
        filename = place.split(',')[0]
    G = ox.graph_from_place(place, network_type="drive")
    G = G.to_undirected()
    ox.plot_graph(G, show=False, save=True, close=True, filepath=f"resources/osm/{filename}.svg")
    ox.save_graphml(G, f"resources/osm/{filename}.graphml")


download("Stockholm")
download("Amsterdam")
download("Zagreb")
download("Oslo")
download("Chisinau")
download("Athens")
download("Helsinki")
download("Kopenhagen")
download("Riga")
download("Vilnius")
