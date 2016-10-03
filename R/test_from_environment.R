# conduct a test from environment
data_rho
rho_list

rho_list_sort <- rho_list[order(rho_list,  decreasing = T)]
for (i in 1:length(rho_list_sort)) {
  if(rho_list_sort[i] < data_rho) {
    break
  }
}
i

