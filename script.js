$(document).ready(function() {
    $('#uploadForm').on('submit', function(e) {
        e.preventDefault();
        
        const fileInput = $('#csvFile')[0];
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        $('#loading').removeClass('d-none');
        $('#results').hide();

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#loading').addClass('d-none');
                $('#results').show();
                
                if (response.success) {
                    const plotsContainer = $('#plots');
                    plotsContainer.empty();
                    
                    response.plots.forEach((plot, index) => {
                        const col = $('<div>').addClass('col-md-6 plot-container');
                        const img = $('<img>').addClass('plot-image')
                            .attr('src', 'data:image/png;base64,' + plot)
                            .attr('alt', 'Plot ' + (index + 1));
                        col.append(img);
                        plotsContainer.append(col);
                    });
                } else {
                    alert('Error: ' + response.error);
                }
            },
            error: function() {
                $('#loading').addClass('d-none');
                alert('An error occurred while processing the file.');
            }
        });
    });
});