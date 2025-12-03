rule flow_metric_plots:
    input:
        metrics = "in/{dataset}.flow_metrics.json.gz"
    output:
        done = "out/{dataset}_plots.done"
    params:
        root = "in/",
        outdir = "out/{dataset}_plots"
    conda:
        "plot_env.yml"
    shell:
        """
        mkdir -p {params.outdir}

        python visualize_flow_metrics.py \
            --root {params.root} \
            --outdir {params.outdir}

        echo 'plots generated' > {output.done}
        """
