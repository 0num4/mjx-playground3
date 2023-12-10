# https://github.com/mjx-project/speed_benchmark/blob/master/run.sh
# speed_benchmark.pyが見つからないので名前を変えている(多分そういうこと？)
N=100
M=5

for agent_type in "tsumogiri" "shanten"; do
    echo "====================================="
    echo ${agent_type}
    echo "====================================="
    for x in $(seq 1 ${M}); do
        echo "run #${x}"
        echo "mjx"
        time python3.9 -O speed_benchmark.py ${N} ${agent_type}
        # echo "mjai"
        # time ./run_mjai.sh ${N} ${agent_type}
    done
done