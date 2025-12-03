rule flow_metric_plots:
    input:
        metrics = "in/{dataset}.flow_metrics.json.gz"
    output:
        plots = "out/{dataset}_plots.done"
    params:
        root = "in/"
    conda:
        "plot_env.yml"
    shell:
        """
        python visualize_flow_metrics.py \
            --root {params.root} \
            --outdir out/{wildcards.dataset}_plots

        # create a dummy flag file to satisfy Snakemake output
        echo "plots generated" > {output.plots}
        """
