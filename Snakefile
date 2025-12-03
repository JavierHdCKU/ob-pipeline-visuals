rule flow_metric_plots:
    input:
        metrics = "{pre}/{post}/{dataset}.flow_metrics.json.gz"
    output:
        done = "{pre}/{post}/plots/flow_metric_plots/{params}/{dataset}_plots.done"
    params:
        # The root directory where your script will search for all metric JSON files
        root = "{pre}/{post}",
        # Where plots will be written
        outdir = "{pre}/{post}/plots/flow_metric_plots/{params}"
    conda:
        "plot_env.yml"
    shell:
        r"""
        echo "== FLOW METRICS PLOTS MODULE =="
        echo "Input metrics: {input.metrics}"
        echo "Root directory: {params.root}"
        echo "Output directory: {params.outdir}"
        echo "Done file: {output.done}"

        mkdir -p {params.outdir}

        python visualize_flow_metrics.py \
            --root {params.root} \
            --outdir {params.outdir}

        echo "plots generated" > {output.done}
        """
