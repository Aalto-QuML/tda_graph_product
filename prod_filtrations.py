import torch
from itertools import product
import itertools


class UnionFind():
    def __init__(self,N):
        self._parents = list(range(0, N))

    def find(self, p):
        return(self._parents[p])

    def merge(self, p, q):
        root_p, root_q = self._parents[p], self._parents[q]
        for i in range(0, len(self._parents)):
            if(self._parents[i] == root_p):
                self._parents[i] = root_q

    def connected(self,p,q):
        return self._parents[p] == self._parents[q]

    def roots(self):
        roots = []
        for i in range(0, len(self._parents)):
            if self._parents[i] == i:
                roots.append(i)
        return roots


def persistence_routine(filtered_v, filtered_e, edge_indices):

    n, m = filtered_v.shape[0], edge_indices.shape[1]

    filtered_e, sorted_indices = torch.sort(filtered_e)
    uf = UnionFind(n)
    persistence = torch.zeros((n, 3), device=filtered_v.device)
    persistence1 = torch.zeros((m, 2), device=filtered_v.device)

    unpaired_value = filtered_e[-1]  # used as infinity

    for edge_index, edge_weight in zip(sorted_indices, filtered_e):

        nodes = edge_indices[:, edge_index]
        younger = uf.find(nodes[0])
        older = uf.find(nodes[1])
        if younger == older:
            persistence1[edge_index, 0] = edge_weight # filtered_e[edge_index]
            persistence1[edge_index, 1] = -1
            continue
        else:
            if filtered_v[younger] < filtered_v[older]:
                younger, older = older, younger
                nodes = torch.flip(nodes, [0])

        persistence[younger, 0] = younger
        persistence[younger, 1] = filtered_v[younger]
        persistence[younger, 2] = edge_weight

        uf.merge(nodes[0], nodes[1])

    for root in uf.roots():
        persistence[root, 0] = root
        persistence[root, 1] = filtered_v[root]
        persistence[root, 2] = -1

    return persistence, persistence1

def thm4_ph(graph_data_item):
  _, edge_index_G, filtered_v_G, filtered_e_G = graph_data_item[0]
  _, edge_index_H, filtered_v_H, filtered_e_H = graph_data_item[1]

  # Precompute the Persistence Pairs using the Union Find Code Above
  impl1 = persistence_routine(filtered_v_G, filtered_e_G, edge_index_G)[0]
  impl2 = persistence_routine(filtered_v_H, filtered_e_H, edge_index_H)[0]

  # Both should have the same filtration steps, this uniformizes their time steps
  filtration_steps = torch.unique(torch.cat((filtered_v_G, filtered_v_H))).tolist()
  filtration_steps.sort()
  filtration_steps.append(-1)

  # Initialize non-trivial deaths list
  G_nd = torch.zeros(len(filtration_steps)+1).tolist()
  H_nd = torch.zeros(len(filtration_steps)+1).tolist()
  G_nd[0] = (0, [])
  H_nd[0] = (0, [])

  # Initialize the trivial deaths list
  G_td = torch.zeros(len(filtration_steps)+1).tolist()
  H_td = torch.zeros(len(filtration_steps)+1).tolist()
  G_td[0] = (0, [])
  H_td[0] = (0, [])

  # Initialize the non-trivial births list
  G_nb = torch.zeros(len(filtration_steps)+1).tolist()
  H_nb = torch.zeros(len(filtration_steps)+1).tolist()
  G_nb[0] = (0, [])
  H_nb[0] = (0, [])

  # Precompute the non-trivial deaths, trivial deaths, and the non-trivial births
  for i in range(0, len(filtration_steps)):
    a_i = filtration_steps[i]

    G_deaths = [int(x[0].item()) for x in impl1 if x[2] == a_i and x[1] != x[2]]
    H_deaths = [int(x[0].item()) for x in impl2 if x[2] == a_i and x[1] != x[2]]
    G_nd[i+1] = (len(G_deaths), G_deaths)
    H_nd[i+1] = (len(H_deaths), H_deaths)

    G_trivial_deaths = [int(x[0].item()) for x in impl1 if x[2] == a_i and x[1] == x[2]]
    H_trivial_deaths = [int(x[0].item()) for x in impl2 if x[2] == a_i and x[1] == x[2]]
    G_td[i+1] = (len(G_trivial_deaths), G_trivial_deaths)
    H_td[i+1] = (len(H_trivial_deaths), H_trivial_deaths)


    G_births = [int(x[0].item()) for x in impl1 if x[1] == a_i and x[1] != x[2]]
    H_births = [int(x[0].item()) for x in impl2 if x[1] == a_i and x[1] != x[2]]

    G_nb[i+1] = (len(G_births), G_births)
    H_nb[i+1] = (len(H_births), H_births)

  # Makes a count of the zeroth Betti number list
  G_b0 = torch.zeros(len(filtration_steps)+1).tolist()
  H_b0 = torch.zeros(len(filtration_steps)+1).tolist()
  G_b0[0] = (0, [])
  H_b0[0] = (0, [])

  for i in range(0, len(filtration_steps)):
    G_count = G_b0[i][0] + G_nb[i+1][0] - G_nd[i+1][0]
    H_count = H_b0[i][0] + H_nb[i+1][0] - H_nd[i+1][0]

    G_list = list(set(G_b0[i][1] + G_nb[i+1][1]).difference(set(G_nd[i+1][1])))
    H_list = list(set(H_b0[i][1] + H_nb[i+1][1]).difference(set(H_nd[i+1][1])))

    G_b0[i+1] = (G_count, G_list)
    H_b0[i+1] = (H_count, H_list)

  # Now we start computing 0-dim Vertex PH of the Product
  nG = filtered_v_G.shape[0]
  nH = filtered_v_H.shape[0]

  # Initialize Persistence
  persistence = []
  for i in range(0, nG):
    input = []
    for j in range(0, nH):
      input.append([i, j, torch.max(filtered_v_G[i], filtered_v_H[j]).item(), None])
    persistence.append(input)

  # Create the non-trivial non-permanent holes
  for i in range(0, len(filtration_steps)):
    a_i = filtration_steps[i]
    g_count, g_deaths = G_nd[i+1]
    h_count, h_deaths = H_nd[i+1]

    _, g_t_deaths = G_td[i+1]
    _, h_t_deaths = H_td[i+1]

    _, g_births = G_nb[i+1]
    _, h_births = H_nb[i+1]

    _, g_betti0 = G_b0[i+1]
    _, h_betti0 = H_b0[i+1]

    # Mark Non-trivial Deaths
    for v in h_deaths:
      for ell in range(0, len(filtered_v_G)):
        item = persistence[ell][v]
        if item[3] == None and (item[2] < a_i or a_i == -1): # ie. the tuple has not been marked dead yet
          item[3] = a_i
    for w in g_deaths:
      for k in range(0, len(filtered_v_H)):
        item = persistence[w][k]
        if item[3] == None and (item[2] < a_i or a_i == -1): # ie. the tuple has not been marked dead yet
          item[3] = a_i

    _, h_betti_prev = H_b0[i]
    _, g_betti_prev = G_b0[i]

    non_trivial_births_current = list(product(g_births, h_betti_prev)) + list(product(g_betti0, h_births))
    non_trivial_deaths_second = list(product(g_births, h_deaths))
    non_trivial_births_current = list(set(non_trivial_births_current).difference(set(non_trivial_deaths_second)))

    # There is an alternative way to compute the non_trivial_births_current as follows:
    # non_trivial_births_current = list(product(g_births, h_betti_prev)) + list(product(g_betti0, h_births))
    # non_trivial_births_current_alt = list(product(g_betti_prev, h_births)) + list(product(g_births, h_betti0))
    # non_trivial_births_current = list(set(non_trivial_births_current).intersection(set(non_trivial_births_current_alt)))

    # Mark Trivial Deaths
    for a in range(0, len(filtered_v_G)):
      for b in range(0, len(filtered_v_H)):
        item = persistence[a][b]
        if item[3] == None and item[2] == a_i and (a, b) not in non_trivial_births_current:
          item[3] = a_i

  # Forget about the labels to the persistence pairs at the end
  projected_persistence = list(itertools.chain(*[[[float(x[2]), float(x[3])] for x in y] for y in persistence]))
  return projected_persistence


# Converting the dataset inputs to entries for the 3 algorithms
# Here we filtrate the two graphs using a degree-based vertex filtration.
def dataset_entry_to_input(item, input_v_G, input_v_H):
  G, H = item
  G_edge_list = [[], []]
  H_edge_list = [[], []]
  for v0, v1 in G.edges:
    G_edge_list[0].append(v0)
    G_edge_list[1].append(v1)

  for v0, v1 in H.edges:
    H_edge_list[0].append(v0)
    H_edge_list[1].append(v1)

  G_nodes = list(G.nodes)
  H_nodes = list(H.nodes)

  # Degree
#  input_v_G = [G.degree[x] for x in G_nodes]
#  input_v_H = [H.degree[x] for x in H_nodes]

  edge_index_G = torch.Tensor(G_edge_list).long()
  filtered_v_G = torch.Tensor(input_v_G)
  filtered_e_G, _ = torch.max(torch.stack((filtered_v_G[edge_index_G[0]], filtered_v_G[edge_index_G[1]])), axis=0)

  edge_index_H = torch.Tensor(H_edge_list).long()
  filtered_v_H = torch.Tensor(input_v_H)
  filtered_e_H, _ = torch.max(torch.stack((filtered_v_H[edge_index_H[0]], filtered_v_H[edge_index_H[1]])), axis=0)

  implG = persistence_routine(filtered_v_G, filtered_e_G, edge_index_G)[0]
  implH = persistence_routine(filtered_v_H, filtered_e_H, edge_index_H)[0]

  return (implG, implH), ((input_v_G, edge_index_G, filtered_v_G, filtered_e_G), (input_v_H, edge_index_H, filtered_v_H, filtered_e_H))