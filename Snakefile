rule flow_metric_plots:
    input:
        metrics = "in/{dataset}.flow_metrics.json.gz"
    output:
        done = "out/{dataset}_plots.done"
    conda:
        "plot_env.yml"
    shell:
        """
        mkdir -p out/plots/{wildcards.dataset}

        python visualize_flow_metrics.py \
            {input.metrics} \
            out/plots/{wildcards.dataset}

        echo "plots generated" > {output.done}
        """
