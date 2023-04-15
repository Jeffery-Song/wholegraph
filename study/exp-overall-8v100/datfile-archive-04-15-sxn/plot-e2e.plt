outputfname = "figure-e2e.svg"
dat_file='data.dat'

col_app                          = 1
col_cache_policy                 = 2
col_cache_percent                = 3
col_batch_size                   = 4
col_dataset                      = 5
col_num_worker                   = 6
col_use_amp                      = 7
col_epoch_time                   = 8
col_cuda_mem                     = 9
col_step_time_sample      = 10
col_step_time_feat_copy   = 11
col_step_time_train_total = 12
col_Time_L                       = 13
col_Time_R                       = 14
col_Time_C                       = 15
col_Wght_L                       = 16
col_Wght_R                       = 17
col_Wght_C                       = 18
# col_Thpt_L                       = 19
# col_Thpt_R                       = 20
# col_Thpt_C                       = 21
# col_SizeGB_L                     = 22
# col_SizeGB_R                     = 23
# col_SizeGB_C                     = 24
# col_optimal_local_rate           = 25
# col_optimal_remote_rate          = 26
# col_optimal_cpu_rate             = 27
# col_coll_cache:local_cache_rate  = 28
# col_coll_cache:remote_cache_rate = 29
# col_coll_cache:global_cache_rate = 30
# col_train_process_time           = 31
# col_epoch_time_train_total       = 32
# col_epoch_time_copy_time         = 33
# col_z = 34

set datafile sep '\t'
set output outputfname

# set terminal svg "Helvetica,16" enhance color dl 2 background rgb "white"
set terminal svg size 1000,900 font "Helvetica,16" enhanced background rgb "white" dl 2
set multiplot layout 3,3

set style fill solid border -2
set pointsize 1
set boxwidth 0.5 relative

set tics font ",12" scale 0.5

set rmargin 1.6
set lmargin 5.5
set tmargin 1.5
set bmargin 2

#### magic to filter expected data entry from dat file
format_str="<awk -F'\\t' '{ if(". \
                              "$".col_app."          ~ \"%s\"     && ". \
                              "$".col_dataset."      ~ \"%s\"     && ". \
                              "$".col_batch_size."   ~ \"%s\"     && ". \
                              "$".col_cache_policy." ~ \"%s\"     ". \
                              ") { print }}' ".dat_file." "
cmd_filter_dat_by_policy(app, dataset, batch_size, policy)=sprintf(format_str, app, dataset, batch_size, policy)
##########################################################################################

### Key
set key inside left Left reverse top enhanced box 
set key samplen 1 spacing 1 height 0.2 width -2 font ',13' maxrows 4 center at graph 0.65, graph 0.6 noopaque

set xlabel "Cache Rate" offset 0,1.7 font ",13"
set xrange [0:]
# set xtics rotate by -45
set xtics nomirror offset 0,0.7

## Y-axis
set ylabel "Copy Time(ms)" offset 3.5,0 font ",13"
set yrange [0:]
# set ytics 0,20,100 
set ytics offset 0.5,0 #format "%.1f" #nomirror

set grid ytics

cache_percent_lb = 0

app_str = "unsup unsup unsup unsup unsup unsup sup  sup  sup"
ds_str  = "PA    PA    PA    CF    CF    MAG   PA   CF   MAG"
bs_str  = "2000  4000  8000  2000  4000  2000  8000 8000 8000"

do for [i=1:words(app_str)]  {
app = word(app_str, i)
ds = word(ds_str, i)
bs = word(bs_str, i)
set title app." ".ds." ".bs offset 0,-1
# print cmd_filter_dat_by_policy("_".app, ds, bs, "^Rep")
plot cmd_filter_dat_by_policy("_".app, ds, bs, "^Rep")          using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_epoch_time))     w lp ps 0.5 lw 1 lc 3 title "Rep" \
    ,cmd_filter_dat_by_policy("_".app, ds, bs, "^Cliq")         using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_epoch_time))     w lp ps 0.5 lw 1 lc 2 title "CliqPart" \
    ,cmd_filter_dat_by_policy("_".app, ds, bs, "^Coll")         using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_epoch_time))     w lp ps 0.5 lw 1 lc 1 title "Coll" \
    ,cmd_filter_dat_by_policy("_".app, ds, bs, "MPSPhaseColl")  using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_epoch_time))     w lp ps 0.5 lw 3 lc 1 title "CollPhase" \
    ,cmd_filter_dat_by_policy("_".app, ds, bs, "WG")            using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_epoch_time))     w lp ps 0.5 lw 1 lc 4 title "WG" \
    # ,cmd_filter_dat_by_policy("_".app, ds, bs, "MPSPhaseRep")   using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_epoch_time))     w lp ps 0.5 lw 3 lc 3 title "MPSRep" \
    # ,cmd_filter_dat_by_policy("_".app, ds, bs, "MPSPhaseCliq")  using (column(col_cache_percent) > cache_percent_lb ? column(col_cache_percent) : 1/0):(column(col_epoch_time))     w lp ps 0.5 lw 3 lc 2 title "MPSCliqPart" \

}
